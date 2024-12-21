#ifndef _BBS_H_
#define _BBS_H_

#include <iostream>
#include <fstream>
#include <map>


#include <eigen3/Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "map.h"
#include "tic_toc.h"

using namespace std;

typedef vector<Eigen::Array2i> DiscreteScan2D;
int bab_max_depth = 5;//进行全局定位时预计算图的深度(数量)
int numberOfCores = 4;
float csm_global_match_lowest_score = 0.8;
float best_solution_lowest_socre = 0.6;
float csm_global_match_angular_resolution = 1.0;
bool  prune_impossible_regions = false;
int   possible_candidates_number = 5;
map_t* map_;//根据点云地图建立的似然域
float occupied_and_unknown_ratio_for_impossible_regions = 0.9;
//预计算一些标志位：0表示“明显”不可能的候选解，1表示可能的候选解
vector<char> candidate_flag_in_lowest_resolution;
vector<vector<int>> lowest_resolution_impossible_regions;

// A possible solution.
struct Candidate {
  Candidate(){}
  Candidate(const int init_scan_index, const int init_x_index_offset, const int init_y_index_offset, const int init_depth): 
            scan_index(init_scan_index), x_index_offset(init_x_index_offset), y_index_offset(init_y_index_offset), depth(init_depth){}

  // Index into the rotated scans vector.
  int scan_index = 0;
  // Linear offset from the initial pose.
  int x_index_offset = 0;
  int y_index_offset = 0;

  int depth = 0;

  // Score, higher is better.
  float score = 0.f;

  //定义了两个比较操作符的重载用于方便比较候选点的优劣。
  bool operator<(const Candidate& other) const { return score < other.score; }
  bool operator>(const Candidate& other) const { return score > other.score; }
};

//滑动窗口类
class SlidingWindowMaximum {
 public:
  //添加窗口覆盖范围内可能的最大值
  void AddValue(const float value) {
    while (!non_ascending_maxima_.empty() &&
           value > non_ascending_maxima_.back()) {
      non_ascending_maxima_.pop_back();
    }
    non_ascending_maxima_.push_back(value);
  }
  //把不属于（已经滑过）窗口覆盖范围内的值删掉
  void RemoveValue(const float value) {
    if (value == non_ascending_maxima_.front()) {
      non_ascending_maxima_.pop_front();
    }
  }

  float GetMaximum() const {
    return non_ascending_maxima_.front();
  }

 private:
  //容器中的得分降序排列
  std::deque<float> non_ascending_maxima_;
};

//分支定界过程中用到的预算图
class PrecomputationGrid2D {
 public:
  PrecomputationGrid2D(const int width);

  //根据原始分辨率下的栅格索引获取当前分辨率预算图下的栅格得分
  float GetValue(const Eigen::Array2i& xy_index) const { 
    const Eigen::Array2i local_xy_index = xy_index - offset_; 
    //负数转为unsigned类型会输出计算机所能表示的最大正整数，参考：https://www.cnblogs.com/foremember/p/10472515.html
    //这两个if里的条件实际代表了四个条件：xy_index.x() < offset_.x() || xy_index.y() < offset_.y() || local_xy_index.x() >= size_x || local_xy_index.y() >= size_y
    if (static_cast<unsigned>(local_xy_index.x()) >= static_cast<unsigned>(size_x) || static_cast<unsigned>(local_xy_index.y()) >= static_cast<unsigned>(size_y)){
        return 0;
    } 
    return cells_[local_xy_index.x() + local_xy_index.y() * size_x];
  }
  const Eigen::Array2i offset_;

  // Size of the precomputation grid.
  int size_x, size_y;

  //当前分辨率预算图下的各栅格得分
  //cartographer用的uint8类型，应该是为了节省内存，但这样会不会损失精度？-2022.4.18
  float* cells_;
};

PrecomputationGrid2D::PrecomputationGrid2D(const int width): 
  offset_(-width + 1, -width + 1), size_x(map_->size_x + width - 1), size_y(map_->size_y + width - 1) {

  cells_ = new float[size_x * size_y];

  int stride = size_x;
  //按行通过滑动窗口取窗口x方向最大值
  vector<float> intermediate;
  //每行多了width-1个栅格
  intermediate.resize(size_x * map_->size_y);
  for (int y = 0; y != map_->size_y; ++y) {
    SlidingWindowMaximum current_values;
    current_values.AddValue(map_->cells[MAP_INDEX(map_,0,y)].score);
    //未完全滑入
    for (int x = -width + 1; x != 0; ++x) {
      intermediate[x + width - 1 + y * stride] = current_values.GetMaximum();
      current_values.AddValue(map_->cells[MAP_INDEX(map_,x+width,y)].score);
    }
    //完全滑入
    for (int x = 0; x < map_->size_x - width; ++x) {
      intermediate[x + width - 1 + y * stride] = current_values.GetMaximum();
      current_values.RemoveValue(map_->cells[MAP_INDEX(map_,x,y)].score);
      current_values.AddValue(map_->cells[MAP_INDEX(map_,x+width,y)].score);
    }
    //滑出
    for (int x = map_->size_x - width;x != map_->size_x; ++x) {
      intermediate[x + width - 1 + y * stride] = current_values.GetMaximum();
      current_values.RemoveValue(map_->cells[MAP_INDEX(map_,x,y)].score);
    }
  }
  //按列通过滑动窗口取窗口y方向最大值，得到的结果即为width宽度窗口内的最大值
  for (int x = 0; x != size_x; ++x) {
    SlidingWindowMaximum current_values;
    current_values.AddValue(intermediate[x]);
    //未完全滑入
    for (int y = -width + 1; y != 0; ++y) {
      cells_[x + (y + width - 1) * stride] = current_values.GetMaximum();
      current_values.AddValue(intermediate[x + (y + width) * stride]);
    }
    //完全滑入
    for (int y = 0; y < map_->size_y - width; ++y) {
      cells_[x + (y + width - 1) * stride] = current_values.GetMaximum();
      current_values.RemoveValue(intermediate[x + y * stride]);
      current_values.AddValue(intermediate[x + (y + width) * stride]);
    }
    //滑出
    for (int y = map_->size_y - width;y != map_->size_y; ++y) {
      cells_[x + (y + width - 1) * stride] = current_values.GetMaximum();
      current_values.RemoveValue(intermediate[x + y * stride]);
    }
  }
  if(prune_impossible_regions && (width == 1 << (bab_max_depth - 1))){
    //计算candidate_flag_in_lowest_resolution
    int x_count = ceil(1.0*map_->size_x/width);
    int y_count = ceil(1.0*map_->size_y/width);
    candidate_flag_in_lowest_resolution.resize(x_count*y_count, 1);
    for(int i = 0; i < x_count; i++){
        for(int j = 0; j < y_count; j++){
          //遍历每一个子区域
          int occ_count = 0;
          int unknown_count = 0;
          int total_count = 0;
          for(int k = i*width; k < (i+1)*width; k++){
            for(int l = j*width; l < (j+1)*width; l++){
              if(MAP_VALID(map_,k,l)){
                total_count++;
                //占据栅格个数
                if(map_->cells[MAP_INDEX(map_,k,l)].occ_state == 1)
                  occ_count++;
                //未知栅格个数
                else if(map_->cells[MAP_INDEX(map_,k,l)].occ_state == 0)
                  unknown_count++;
              }
            }
          }
          if((occ_count+unknown_count) > occupied_and_unknown_ratio_for_impossible_regions*total_count){
            candidate_flag_in_lowest_resolution[i+j*x_count] = 0;
            //记录最低分辨率下的不可能区域，方便后面可视化。格式：左上角顶点x，左上角顶点y，边长，边长
            if(map_->size_y-1-(j+1)*width >= 0 && (i+1)*width < map_->size_x)
              lowest_resolution_impossible_regions.push_back({i*width, map_->size_y-1-(j+1)*width, width, width});
          }
        }
      }
  }
}

class BBS_CSM
{
public:
    float initial_x;
    float initial_y;
    int x_linear_search_bound;
    int y_linear_search_bound;
    int cloudSize;//当前帧激光点云中的点数量
    float angular_resolution_radian;
    int rotated_num;
    vector<Candidate> lowest_resolution_candidates;
    vector<int> scored_nodes_number_in_each_layer;
    double total_time_for_score_candidates;

    //多分辨率预算图
    vector<PrecomputationGrid2D> precomputation_grids_;

    BBS_CSM(const float initial_x_, const float initial_y_, const int x_linear_search_bound_, const int y_linear_search_bound_):
            initial_x(initial_x_), initial_y(initial_y_), x_linear_search_bound(x_linear_search_bound_), y_linear_search_bound(y_linear_search_bound_) {

        angular_resolution_radian = csm_global_match_angular_resolution/180.0*M_PI;
        rotated_num = int(360/csm_global_match_angular_resolution);

        scored_nodes_number_in_each_layer.resize(bab_max_depth, 0);
        total_time_for_score_candidates = 0.0;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr transformPointCloud(float curr_heading, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudIn){
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOut(new pcl::PointCloud<pcl::PointXYZ>());
        cloudOut->resize(cloudSize);
        float cosx = cos(curr_heading);
        float sinx = sin(curr_heading);
        //这里不用多线程了，因为一帧激光数据的点数比较少，频繁启动线程会带来额外的开销，因为线程需要被创建和销毁
        //#pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < cloudSize; ++i)
        {
            const auto &pointFrom = cloudIn->points[i];
            cloudOut->points[i].x = cosx * pointFrom.x - sinx * pointFrom.y ;
            cloudOut->points[i].y = sinx * pointFrom.x + cosx * pointFrom.y ;
            cloudOut->points[i].z = 0;
        }
        return cloudOut;
    }

    vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> generateRotatedScans(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){
        vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> rotated_scans;
        //reserve只分配内存，不创建对象
        rotated_scans.reserve(rotated_num);
        float delta_theta = -M_PI;
        for(int i = 0; i < rotated_num; i++){
            rotated_scans.push_back(transformPointCloud(delta_theta,cloud));
            delta_theta += angular_resolution_radian;
        }
        return rotated_scans;
    }

    void globalMatch_DFS(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloudInBaseFrame, vector<float>& best_solution, int& total_number){
        cloudSize = cloudInBaseFrame->size();
        //计算按角度分辨率和搜索窗口(PI)旋转后的点云集合
        vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> rotated_scans = generateRotatedScans(cloudInBaseFrame);
        //将旋转后的点云集合加入初值平移量转换到地图坐标下，并将各个激光点坐标转换成栅格索引
        const vector<DiscreteScan2D> discrete_scans = DiscretizeScans(rotated_scans);
        //用局部变量访问速度更快，参考：https://www.w3cschool.cn/article/37805725.html
        vector<Candidate> tem(lowest_resolution_candidates.begin(), lowest_resolution_candidates.end());
        ScoreCandidates(discrete_scans, &tem, bab_max_depth-1);
        //按评分从大到小排序
        vector<Candidate> candidates;
        for (const auto& candidate : tem) {
          if (candidate.score < csm_global_match_lowest_score)
            continue;
          candidates.push_back(candidate);
        }
        std::sort(candidates.begin(), candidates.end(), std::greater<Candidate>());
        //分支定界搜索
        const Candidate best_candidate = BranchAndBound(discrete_scans, candidates, bab_max_depth-1, best_solution_lowest_socre);
        float best_x = best_candidate.x_index_offset*map_->scale + initial_x;
        float best_y = best_candidate.y_index_offset*map_->scale + initial_y;
        float best_yaw = best_candidate.scan_index*angular_resolution_radian-M_PI;
        best_solution = {best_x, best_y, best_yaw};
        total_number = 0;
        for(int i = 0; i < bab_max_depth; i++){
          total_number += scored_nodes_number_in_each_layer[i];
        }
        fill(scored_nodes_number_in_each_layer.begin(), scored_nodes_number_in_each_layer.end(), 0);
    }

    void globalMatch_BFS(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloudInBaseFrame, vector<float>& best_solution, int& total_number){
      cloudSize = cloudInBaseFrame->size();
      //计算按角度分辨率和搜索窗口(PI)旋转后的点云集合
      vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> rotated_scans = generateRotatedScans(cloudInBaseFrame);
      //将旋转后的点云集合加入初值平移量转换到地图坐标下，并将各个激光点坐标转换成栅格索引
      const vector<DiscreteScan2D> discrete_scans = DiscretizeScans(rotated_scans);
      //用局部变量访问速度更快，参考：https://www.w3cschool.cn/article/37805725.html
      vector<Candidate> tem(lowest_resolution_candidates.begin(), lowest_resolution_candidates.end());
      ScoreCandidates(discrete_scans, &tem, bab_max_depth-1); 
      priority_queue<Candidate> candidates_queue;
      for (const auto& candidate : tem) {
        if (candidate.score < csm_global_match_lowest_score)
          continue;
        candidates_queue.push(candidate);
      }
      //叶子节点的评分不能低于该值
      float best_solution_socre = best_solution_lowest_socre;
      Candidate best_candidate;
      while(!candidates_queue.empty()){
        Candidate this_candidate = candidates_queue.top();
        candidates_queue.pop();
        if(this_candidate.score < best_solution_socre)
          continue;
        if(this_candidate.depth == 0){
          best_solution_socre = this_candidate.score;
          best_candidate = this_candidate;
        }
        else{
          //分支
          vector<Candidate> higher_resolution_candidates;
          const int half_width = 1 << (this_candidate.depth - 1);
          for (int x_offset : {0, half_width}) {
            if (this_candidate.x_index_offset + x_offset > x_linear_search_bound) {
              break;
            }
            for (int y_offset : {0, half_width}) { 
              if (this_candidate.y_index_offset + y_offset > y_linear_search_bound) {
                break;
              }    
              higher_resolution_candidates.emplace_back(this_candidate.scan_index, this_candidate.x_index_offset + x_offset, this_candidate.y_index_offset + y_offset, this_candidate.depth-1);
            }
          }
          ScoreCandidates(discrete_scans, &higher_resolution_candidates, this_candidate.depth-1);
          for (const auto& child : higher_resolution_candidates) {
            // pruning
            if (child.score < best_solution_socre)
              continue;
            candidates_queue.push(child);
          }
        }
      }
      float best_x = best_candidate.x_index_offset*map_->scale + initial_x;
      float best_y = best_candidate.y_index_offset*map_->scale + initial_y;
      float best_yaw = best_candidate.scan_index*angular_resolution_radian-M_PI;
      best_solution = {best_x, best_y, best_yaw};
      total_number = 0;
      for(int i = 0; i < bab_max_depth; i++){
        total_number += scored_nodes_number_in_each_layer[i];
      }
      fill(scored_nodes_number_in_each_layer.begin(), scored_nodes_number_in_each_layer.end(), 0);
  }

  void globalMatch_CBFS(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloudInBaseFrame, vector<float>& best_solution, int& total_number){
      cloudSize = cloudInBaseFrame->size();
      //计算按角度分辨率和搜索窗口(PI)旋转后的点云集合
      vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> rotated_scans = generateRotatedScans(cloudInBaseFrame);
      //将旋转后的点云集合加入初值平移量转换到地图坐标下，并将各个激光点坐标转换成栅格索引
      const vector<DiscreteScan2D> discrete_scans = DiscretizeScans(rotated_scans);
      //用局部变量访问速度更快，参考：https://www.w3cschool.cn/article/37805725.html
      vector<Candidate> tem(lowest_resolution_candidates.begin(), lowest_resolution_candidates.end());
      ScoreCandidates(discrete_scans, &tem, bab_max_depth-1);
      priority_queue<Candidate> highest_contour;
      for (const auto& candidate : tem) {
        if (candidate.score < csm_global_match_lowest_score)
          continue;
        highest_contour.push(candidate);
      }
      int highest_contour_size = highest_contour.size();
      if(highest_contour_size == 0){
        printf("Error, size of highest contour is %d\n", highest_contour_size);
        best_solution = {initial_x, initial_y, 0};
        return;
      }
      //
      //叶子节点的评分不能低于该值
      float best_solution_socre = best_solution_lowest_socre;
      Candidate best_candidate;
      int current_contour_index = bab_max_depth-1;
      map<int, priority_queue<Candidate>> all_solutions = {{current_contour_index, highest_contour}};
      while(!all_solutions.empty()){
        //更新contour索引
        //rbegin()访问map中最后一组pair
        if(current_contour_index >= all_solutions.rbegin()->first)
          current_contour_index = all_solutions.begin()->first;
        else{
          for(auto iter = all_solutions.begin(); iter != all_solutions.end(); iter++){
            if(iter->first > current_contour_index){
              current_contour_index = iter->first;
              break;
            }
          }
        }
        Candidate this_candidate = all_solutions[current_contour_index].top();
        all_solutions[current_contour_index].pop();
        if(all_solutions[current_contour_index].empty())
          all_solutions.erase(current_contour_index);
        if(this_candidate.score > best_solution_socre){
          if(this_candidate.depth == 0){
            best_solution_socre = this_candidate.score;
            best_candidate = this_candidate;
          }
          else{
            //分支
            vector<Candidate> higher_resolution_candidates;
            const int half_width = 1 << (this_candidate.depth - 1);
            for (int x_offset : {0, half_width}) {
              if (this_candidate.x_index_offset + x_offset > x_linear_search_bound) {
                break;
              }
              for (int y_offset : {0, half_width}) { 
                if (this_candidate.y_index_offset + y_offset > y_linear_search_bound) {
                  break;
                }    
                higher_resolution_candidates.emplace_back(this_candidate.scan_index, this_candidate.x_index_offset + x_offset, this_candidate.y_index_offset + y_offset, this_candidate.depth-1);
              }
            }
            ScoreCandidates(discrete_scans, &higher_resolution_candidates, this_candidate.depth-1);
            for (const auto& child : higher_resolution_candidates) {
              // pruning
              if (child.score < best_solution_socre)
                continue;
              all_solutions[current_contour_index-1].push(child);
            }
          }
        }
      }
      float best_x = best_candidate.x_index_offset*map_->scale + initial_x;
      float best_y = best_candidate.y_index_offset*map_->scale + initial_y;
      float best_yaw = best_candidate.scan_index*angular_resolution_radian-M_PI;
      best_solution = {best_x, best_y, best_yaw};
      total_number = 0;
      for(int i = 0; i < bab_max_depth; i++){
        total_number += scored_nodes_number_in_each_layer[i];
      }
      fill(scored_nodes_number_in_each_layer.begin(), scored_nodes_number_in_each_layer.end(), 0);
  }

  vector<PrecomputationGrid2D> generatePrecomputationGrid(){
      precomputation_grids_.reserve(bab_max_depth);
      for (int i = 0; i != bab_max_depth; ++i) {
          const int width = 1 << i;
          precomputation_grids_.emplace_back(width);
      }
      return precomputation_grids_;
  }

  vector<DiscreteScan2D> DiscretizeScans(vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& scans){
      vector<DiscreteScan2D> discrete_scans;
      discrete_scans.reserve(rotated_num);
      for (int i = 0; i != rotated_num; ++i) {
          discrete_scans.emplace_back();
          discrete_scans.back().reserve(scans[i]->points.size());
          for (int j = 0; j != int(scans[i]->points.size()); ++j) {
            float this_x = scans[i]->points[j].x + initial_x;
            float this_y = scans[i]->points[j].y + initial_y;
            int mi = MAP_GXWX(map_, this_x);
            int mj = MAP_GYWY(map_, this_y);
            discrete_scans.back().emplace_back(mi, mj);
            //这里不要直接把宏定义函数写到emplace_back中，如下面的写法，这样y方向的数值会出错，目前不清楚为什么会出错。-2022.4.6
            //discrete_scans.back().emplace_back(MAP_GXWX(map_, this_x), MAP_GXWX(map_, this_y));
          }
      }
      //ROS_INFO("Invalid scan points number is:%d, all scan points number is:%d",count, cloudSize*int(scans.size()));
      return discrete_scans;
  }

  void GenerateLowestResolutionCandidates(){
    //步长
    const int linear_step_size = 1 << (bab_max_depth - 1);
    //保证至少有一个候选解，因为在位姿跟踪中搜索窗口较小，如果多分辨率地图层数太多，有可能低分辨率栅格步长大于搜索窗口
    const int num_lowest_resolution_linear_x_candidates = (x_linear_search_bound*2 + linear_step_size) / linear_step_size;
    const int num_lowest_resolution_linear_y_candidates = (y_linear_search_bound*2 + linear_step_size) / linear_step_size;
    int num_candidates = num_lowest_resolution_linear_x_candidates*num_lowest_resolution_linear_y_candidates*rotated_num;
    //构建最低分辨率下的候选解集合
    lowest_resolution_candidates.reserve(num_candidates);
    if(prune_impossible_regions){
      for (int scan_index = 0; scan_index != rotated_num; ++scan_index) {
        for (int x_index_offset = -x_linear_search_bound; x_index_offset <= x_linear_search_bound; x_index_offset += linear_step_size) {
          for (int y_index_offset = -y_linear_search_bound; y_index_offset <= y_linear_search_bound; y_index_offset += linear_step_size) {
            int x = (x_index_offset+x_linear_search_bound)/linear_step_size;
            int y = (y_index_offset+y_linear_search_bound)/linear_step_size;
            if(candidate_flag_in_lowest_resolution[x+y*num_lowest_resolution_linear_x_candidates] == 1){
              lowest_resolution_candidates.emplace_back(scan_index, x_index_offset, y_index_offset, bab_max_depth-1);
            }
          }
        }
      }
      lowest_resolution_candidates.shrink_to_fit();
      printf("Lowest resolution candidates theoretical size:%d, possible size:%d\n",num_candidates, int(lowest_resolution_candidates.size()));
    }
    else{
      for (int scan_index = 0; scan_index != rotated_num; ++scan_index) {
        for (int x_index_offset = -x_linear_search_bound; x_index_offset <= x_linear_search_bound; x_index_offset += linear_step_size) {
          for (int y_index_offset = -y_linear_search_bound; y_index_offset <= y_linear_search_bound; y_index_offset += linear_step_size) {
            lowest_resolution_candidates.emplace_back(scan_index, x_index_offset, y_index_offset, bab_max_depth-1);
          }
        }
      }
    }
  }

  void ScoreCandidates(const std::vector<DiscreteScan2D>& discrete_scans, vector<Candidate>* candidates, int depth){
    if(depth == bab_max_depth-1){
      #pragma omp parallel for num_threads(numberOfCores)
      for (int i = 0; i < int(candidates->size()); i++){
          Candidate& candidate = (*candidates)[i];
          float sum = 0;
          for (const Eigen::Array2i& xy_index : discrete_scans[candidate.scan_index]){
              const Eigen::Array2i proposed_xy_index(xy_index.x() + candidate.x_index_offset, xy_index.y() + candidate.y_index_offset);
              sum += precomputation_grids_[depth].GetValue(proposed_xy_index);
          }
          candidate.score = sum / cloudSize;
      }
    }
    else{
      //除了最高层，其他层不再使用openmp，因为调用一次最多也就4个子节点，频繁启动线程会带来额外的开销，因为线程需要被创建和销毁
      //#pragma omp parallel for num_threads(numberOfCores)
      for (int i = 0; i < int(candidates->size()); i++){
          Candidate& candidate = (*candidates)[i];
          float sum = 0;
          for (const Eigen::Array2i& xy_index : discrete_scans[candidate.scan_index]){
              const Eigen::Array2i proposed_xy_index(xy_index.x() + candidate.x_index_offset, xy_index.y() + candidate.y_index_offset);
              sum += precomputation_grids_[depth].GetValue(proposed_xy_index);
          }
          candidate.score = sum / cloudSize;
      }
    }
    scored_nodes_number_in_each_layer[depth] += int(candidates->size());
  }

  Candidate BranchAndBound(const vector<DiscreteScan2D>& discrete_scans, vector<Candidate>& candidates, int candidate_depth, float min_score){
    //递归终止条件
    if (candidate_depth == 0) {
      // 返回最优候选解
      return *candidates.begin();
    }
    //创建一个临时的候选解对象best_high_resolution_candidate，为之赋予最小的评分
    Candidate best_high_resolution_candidate(0,0,0,0);
    for (int k = 0; k < int(candidates.size()); k++) {
      //如果遇到一个候选解的评分很低，意味着以后的候选解中也没有合适的解（因为candidate按score降序排列），直接跳出循环（剪枝）
      if (candidates[k].score <= min_score) {
        break;
      }
      //如果for循环能够继续运行，说明当前候选解是一个更优的选择，需要对其进行分支。新生成的候选点将被保存在容器higher_resolution_candidates中。
      vector<Candidate> higher_resolution_candidates;
      //分支后的步长
      const int half_width = 1 << (candidate_depth - 1);
      for (int x_offset : {0, half_width}) {
        if (candidates[k].x_index_offset + x_offset > x_linear_search_bound) {
          break;
        }
        for (int y_offset : {0, half_width}) { 
          if (candidates[k].y_index_offset + y_offset > y_linear_search_bound) {
            break;
          }    
          higher_resolution_candidates.emplace_back(candidates[k].scan_index, candidates[k].x_index_offset + x_offset, candidates[k].y_index_offset + y_offset, candidate_depth-1);
        }
      }
      //调用函数ScoreCandidates对新分支的候选解评分并排序，并递归调用BranchAndBound对新分支的
      //higher_resolution_candidates进行搜索。 这样就可以实现深度优先的搜索，先一直沿着最有可能的分支向下搜索，
      //直到找到一个解。并将该解作为目前的最优解保存在best_high_resolution_candidate中。 
      //以后通过递归调用发现了更优的解都将通过std::max函数来更新已知的最优解。
      
      ScoreCandidates(discrete_scans, &higher_resolution_candidates, candidate_depth-1);
      sort(higher_resolution_candidates.begin(), higher_resolution_candidates.end(), std::greater<Candidate>());

      //vector<Candidate> actual_higher_resolution_candidates = {higher_resolution_candidates[0]};
      
      best_high_resolution_candidate = std::max(best_high_resolution_candidate,
                                                BranchAndBound(discrete_scans, higher_resolution_candidates, candidate_depth - 1, best_high_resolution_candidate.score));
    }
    //当遍历完所有的候选点之后，对象best_high_resolution_candidate中就记录了最优的解，返回。
    return best_high_resolution_candidate;
  }

};

#endif