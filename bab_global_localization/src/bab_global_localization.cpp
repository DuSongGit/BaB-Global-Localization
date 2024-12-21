
#include "BBS.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>

#include "yaml-cpp/yaml.h"

#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>

using namespace cv;

Mat map_pgm;
Mat map_mat;
double resolution;//m
double likelihood_max_dist = 0.5;//计算似然域时向障碍物四周膨胀的最远距离[m]
double laser_standard_deviation = 0.2;//计算hit点概率的高斯标准差[m]
bool   show_map_and_likelihood_field = false;
bool   use_voxel_filter = false;
bool   show_impossible_regions_in_lowest_resolution = false;
int    search_strategy = 0;
bool   visual_supervision = false;

void draw_likelihood_field(){
    Mat likelihood_field = Mat(map_->size_y, map_->size_x, CV_8UC1, cv::Scalar(0));
    for(int j = 0; j < map_->size_y; j++){
        for(int i = 0; i < map_->size_x; i++){
            //只需要更改距障碍物距离小于max_occ_dist栅格的像素值
            if(map_->cells[MAP_INDEX(map_,i,j)].occ_dist < map_->max_occ_dist)
                //区域位置越亮，似然度越高(注意：opencv图像坐标系原点在图片左上角，我们提供的map坐标系原点在左下角)
                //likelihood_field.at<uchar>(map_->size_y-1-j,i) = 255*(1 - map_->cells[MAP_INDEX(map_,i,j)].occ_dist/map_->max_occ_dist);
                likelihood_field.at<uchar>(map_->size_y-1-j,i) = 255*map_->cells[MAP_INDEX(map_,i,j)].score;
        }
    }
    imshow("likelihood_field", likelihood_field);
}

void loadMap(string path){
    map_pgm = imread(path+"map.pgm", cv::COLOR_BGR2GRAY);
    YAML::Node doc = YAML::LoadFile(path+"map.yaml");
    double origin_x = doc["origin"][0].as<double>(); //origin_x
    double origin_y = doc["origin"][1].as<double>(); //origin_y
    resolution = doc["resolution"].as<double>();
    //printf("origin x: %f, origin y: %f, resolution: %f\n.", origin_x, origin_y, resolution);
    //构造似然域模型
    map_ = map_alloc();
    map_->size_x = map_pgm.cols;
    map_->size_y = map_pgm.rows;
    map_->scale = resolution;
    //这里把map_原点设为了图片的中心位置，与map.h中定义的地图坐标系与世界坐标系的转换关系相对应
    map_->origin_x = origin_x + (map_->size_x / 2) * map_->scale;
    map_->origin_y = origin_y + (map_->size_y / 2) * map_->scale;
    map_->cells = (map_cell_t*)malloc(sizeof(map_cell_t)*map_->size_x*map_->size_y);
    map_mat = Mat(map_pgm.rows, map_pgm.cols, CV_8UC1, cv::Scalar(0));
    for(int row = 0; row < map_pgm.rows; row++){
        for(int col = 0; col < map_pgm.cols; col++){
            uchar pixel_value = map_pgm.at<uchar>(row, col);
            if(pixel_value == 0){
                map_->cells[MAP_INDEX(map_, col, map_pgm.rows-1-row)].occ_state = +1;
                map_mat.at<uchar>(row, col) = (uchar)255;
            }
            else if(pixel_value == 254)
                map_->cells[MAP_INDEX(map_, col, map_pgm.rows-1-row)].occ_state = -1;
            else
                map_->cells[MAP_INDEX(map_, col, map_pgm.rows-1-row)].occ_state = 0;
        }
    }
    //生成似然域模型
    map_update_cspace(map_ , likelihood_max_dist, laser_standard_deviation);
    if(show_map_and_likelihood_field){
        draw_likelihood_field();
        imshow("pgm map", map_pgm);
        waitKey(0);
        destroyAllWindows();
    }
}

void show_impossible_regions(){
    Mat map_color;
    cvtColor(map_pgm, map_color, COLOR_GRAY2BGR);
    if(int(lowest_resolution_impossible_regions.size()) > 0)
        for(int i = 0; i < int(lowest_resolution_impossible_regions.size()); i++){
            Rect rec1(lowest_resolution_impossible_regions[i][0],lowest_resolution_impossible_regions[i][1],
                      lowest_resolution_impossible_regions[i][2],lowest_resolution_impossible_regions[i][3]);
            rectangle(map_color,rec1,Scalar(0,0,255),1,1,0);
        }
    imshow("impossible_regions_in_lowest_resolution", map_color);
    waitKey(0);
    destroyAllWindows();
}

pcl::PointCloud<pcl::PointXYZ>::Ptr transformPointCloud(float x, float y, float curr_heading, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudIn){
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOut(new pcl::PointCloud<pcl::PointXYZ>());
    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);
    float cosx = cos(curr_heading);
    float sinx = sin(curr_heading);
    #pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i){
        const auto &pointFrom = cloudIn->points[i];
        cloudOut->points[i].x = cosx * pointFrom.x - sinx * pointFrom.y + x;
        cloudOut->points[i].y = sinx * pointFrom.x + cosx * pointFrom.y + y;
        cloudOut->points[i].z = 0;
    }
    return cloudOut;
}

int main(int argc, char** argv)
{
    // 打开配置文件
    ifstream config_file("../config/parameters.txt");
    if (!config_file.is_open())
    {
        cerr << "Failed to open config file." << endl;
        return -1;
    }
    // 逐行读取并解析数据
    string line;
    while (getline(config_file, line))
    {
        // 如果是注释或空行，则忽略
        if (line.empty() || line[0] == '#')
            continue;
        // 解析key-value对
        size_t pos = line.find('=');
        if (pos != string::npos)
        {
            if(line.substr(0, pos) == "bab_max_depth")
                bab_max_depth = stoi(line.substr(pos + 1));
            else if(line.substr(0, pos) == "numberOfCores")
                numberOfCores = stoi(line.substr(pos + 1));
            else if(line.substr(0, pos) == "csm_global_match_lowest_score")
                csm_global_match_lowest_score = stof(line.substr(pos + 1));
            else if(line.substr(0, pos) == "best_solution_lowest_socre")
                best_solution_lowest_socre = stof(line.substr(pos + 1));
            else if(line.substr(0, pos) == "csm_global_match_angular_resolution")
                csm_global_match_angular_resolution = stof(line.substr(pos + 1));
            else if(line.substr(0, pos) == "possible_candidates_number")
                possible_candidates_number = stoi(line.substr(pos + 1));
            else if(line.substr(0, pos) == "use_voxel_filter"){
                if(line.substr(pos+1, 4) == "true")
                    use_voxel_filter = true;             
            }
            else if(line.substr(0, pos) == "show_map_and_likelihood_field"){
                if(line.substr(pos+1, 4) == "true")
                    show_map_and_likelihood_field = true;             
            }
            else if(line.substr(0, pos) == "show_impossible_regions_in_lowest_resolution"){
                if(line.substr(pos+1, 4) == "true")
                    show_impossible_regions_in_lowest_resolution = true;             
            }
            else if(line.substr(0, pos) == "prune_impossible_regions"){
                if(line.substr(pos+1, 4) == "true")
                    prune_impossible_regions = true;             
            }
            else if(line.substr(0, pos) == "occupied_and_unknown_ratio_for_impossible_regions")
                occupied_and_unknown_ratio_for_impossible_regions = stof(line.substr(pos + 1));
            else if(line.substr(0, pos) == "use_likelihood_model_exclude_unknown"){
                if(line.substr(pos+1, 4) == "true")
                    use_likelihood_model_exclude_unknown = true;             
            }
            else if(line.substr(0, pos) == "laser_standard_deviation")
                laser_standard_deviation = stod(line.substr(pos + 1));
            else if(line.substr(0, pos) == "search_strategy"){
                search_strategy = stoi(line.substr(pos + 1));             
            }
            else if(line.substr(0, pos) == "visual_supervision"){
                if(line.substr(pos+1, 4) == "true")
                    visual_supervision = true;             
            }
        }
    }
    // 关闭文件
    config_file.close();

    if (argc < 2)
    {
        std::cout << "Please specify the dataset path!" << std::endl;
        return 1;
    }
    //这里要转换一下，不能直接用argv[1]去和map.pcd相加
    string path = argv[1];
    //加载地图
    loadMap(path);
    int totol_number;
    int step = 1;
    float resize_factor = 1.0;
    vector<int> false_index;
    if(path == "simulation_factory_original/"){
        totol_number = 154;
        resize_factor = 0.6;
        false_index = {150};
    }
    else if(path == "simulation_factory_changed/"){
        totol_number = 135;
        resize_factor = 0.6;
        false_index = {116};
    }
    else if(path == "intel/")
        totol_number = 135;
    else if(path == "aces/"){
        totol_number = 74;
        resize_factor = 0.6;
    }
    else if(path == "corridor/"){
        totol_number = 360;
        step = 2;
        resize_factor = 0.6;
        false_index = {26, 96, 100, 106, 122, 224, 234, 238, 242, 288, 290, 326};
    }
    else if(path == "workshop_2/"){
        totol_number = 1169;
        step = 5;
        resize_factor = 0.4;
    }
    else if(path == "workshop_1"){
        totol_number = 480;
        step = 3;
        resize_factor = 0.6;
    }
    else if(path == "workshop_1_changed/"){
        totol_number = 192;
        resize_factor = 0.6;
        false_index = {104, 106, 107, 108, 109, 110, 111, 112, 114};
    }
    
    float initial_x = MAP_WXGX(map_,map_->size_x/2);
    float initial_y = MAP_WYGY(map_,map_->size_y/2);
    int   x_linear_search_bound = map_->size_x/2;
    int   y_linear_search_bound = map_->size_y/2;
    BBS_CSM bbs_csm(initial_x, initial_y, x_linear_search_bound, y_linear_search_bound);
    //根据似然域模型生成预算图
    bbs_csm.generatePrecomputationGrid();
    //生成最低分辨率候选解
    bbs_csm.GenerateLowestResolutionCandidates();
    if(prune_impossible_regions && show_impossible_regions_in_lowest_resolution)
        show_impossible_regions();
    //体素滤波
    pcl::VoxelGrid<pcl::PointXYZ> downSampleScan;
    downSampleScan.setLeafSize(resolution, resolution, resolution);
    cout << "input start scan index:";
    int scan_index;
    cin >> scan_index;
    for(; scan_index < totol_number; scan_index += step){
        pcl::PointCloud<pcl::PointXYZ>::Ptr scan_pointcloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::io::loadPCDFile(path+"scans/full"+to_string(scan_index) + ".pcd", *scan_pointcloud);
        //全局搜索
        if(use_voxel_filter){
            printf("Size of original scan points: %d\n", int(scan_pointcloud->size()));
            downSampleScan.setInputCloud(scan_pointcloud);
            downSampleScan.filter(*scan_pointcloud);
            printf("Size of filtered scan points: %d\n", int(scan_pointcloud->size()));
        }
        int total_evaluated_number;
        double time_cost;
        vector<float> best_solution;//x, y, rotation
        //dfs
        if(search_strategy == 0){
            TicToc t_register;
            bbs_csm.globalMatch_DFS(scan_pointcloud, best_solution, total_evaluated_number);
            time_cost = t_register.toc();
        }
        //bfs
        else if(search_strategy == 1){
            TicToc t_register;
            bbs_csm.globalMatch_BFS(scan_pointcloud, best_solution, total_evaluated_number);
            time_cost = t_register.toc();
        }
        //cbfs
        else if(search_strategy == 2){
            TicToc t_register;
            bbs_csm.globalMatch_CBFS(scan_pointcloud, best_solution, total_evaluated_number);
            time_cost = t_register.toc();
        }
        printf("\033[1;31mPosition %d global pose: x:%fm, y: %fm, rotation: %fdegrees, time cost: %fms.\033[0m\n", scan_index, best_solution[0], best_solution[1], best_solution[2]/M_PI*180, time_cost);

        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_scan = transformPointCloud(best_solution[0], best_solution[1], best_solution[2], scan_pointcloud);
        Mat scan_mat = Mat(map_mat.rows, map_mat.cols, CV_8UC1, cv::Scalar(0));
        //画图
        for(int i = 0; i < int(transformed_scan->points.size()); i++){
            int map_x = MAP_GXWX(map_, transformed_scan->points[i].x);
            int map_y = MAP_GYWY(map_, transformed_scan->points[i].y);
            if(MAP_VALID(map_, map_x, map_y))
                scan_mat.at<uchar>(scan_mat.rows-1-map_y, map_x) = (uchar)255;
        }
        Mat overlay_image;
        addWeighted(map_mat, 0.2, scan_mat, 0.8, 0.0, overlay_image);
        Mat resized_mat;
        resize(overlay_image, resized_mat, Size(), resize_factor, resize_factor, INTER_LINEAR);
        imshow("overlay_image", resized_mat);
        waitKey(0);
        destroyAllWindows();
        printf("------------------------------------------------------\n");
    }

    if( map_ != NULL ){
        map_free( map_ );
        map_ = NULL;
    }
    return 0;
}
