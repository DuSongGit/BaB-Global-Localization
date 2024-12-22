#ifndef _MAP_H_
#define _MAP_H_

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <queue>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Compute the cell index for the given map coords.
#define MAP_INDEX(map, i, j) ((i) + (j) * map->size_x)
// Convert from map index to world coords
#define MAP_WXGX(map, i) (map->origin_x + ((i) - map->size_x / 2) * map->scale)
#define MAP_WYGY(map, j) (map->origin_y + ((j) - map->size_y / 2) * map->scale)
// Convert from world coords to map coords
#define MAP_GXWX(map, x) (floor((x - map->origin_x) / map->scale + 0.5) + map->size_x / 2)
#define MAP_GYWY(map, y) (floor((y - map->origin_y) / map->scale + 0.5) + map->size_y / 2)
// Test to see if the given map coords lie within the absolute map bounds.
#define MAP_VALID(map, i, j) ((i >= 0) && (i < map->size_x) && (j >= 0) && (j < map->size_y))

double z_hit_denominator;
bool use_likelihood_model_exclude_unknown = false;

// Description for a single map cell.
typedef struct
{
  // Occupancy state (-1 = free, 0 = unknown, +1 = occ)
  int occ_state;
  // Distance to the nearest occupied cell[m]
  double occ_dist;
  float score;
} map_cell_t;

// Description for a map
typedef struct
{
  // Map origin; the map is a viewport onto a conceptual larger map. 图片中心在map坐标系下的位置
  double origin_x, origin_y;
  // Map scale (m/cell)
  double scale;
  // Map dimensions (number of cells)
  int size_x, size_y;
  // The map data, stored as a grid
  map_cell_t *cells;
  // Max distance at which we care about obstacles, for constructing likelihood field
  double max_occ_dist;
  float min_score;
} map_t;

// Create a new map
map_t *map_alloc(void)
{
  map_t *map;
  map = (map_t*) malloc(sizeof(map_t));
  map->origin_x = 0;
  map->origin_y = 0;
  map->size_x = 0;
  map->size_y = 0;
  map->scale = 0;
  map->cells = (map_cell_t*) NULL;
  return map;
}

// Destroy a map
void map_free(map_t *map)
{
  free(map->cells);
  free(map);
  return;
}

//似然域包含的栅格属性
class CellData
{
  public:
    map_t* map_;
    unsigned int i_, j_;//似然域包含的某一个栅格的坐标
    unsigned int src_i_, src_j_;//地图中距离当前栅格最近的障碍物栅格坐标
};

class CachedDistanceMap
{
  public:
    CachedDistanceMap(double scale, double max_dist) : 
      distances_(NULL), scale_(scale), max_dist_(max_dist) 
    {
      cell_radius_ = max_dist / scale;
      distances_ = new double *[cell_radius_+2];
      for(int i=0; i<=cell_radius_+1; i++)
      {
	      distances_[i] = new double[cell_radius_+2];
        for(int j=0; j<=cell_radius_+1; j++)
	      {
	        distances_[i][j] = sqrt(i*i + j*j);
	      }
      }
    }
    ~CachedDistanceMap()
    {
      if(distances_)
      {
	      for(int i=0; i<=cell_radius_+1; i++)
	        delete[] distances_[i];
	      delete[] distances_;
      }
    }
    double** distances_;
    double scale_;
    double max_dist_;
    int cell_radius_;
};

bool operator<(const CellData& a, const CellData& b)
{
  return a.map_->cells[MAP_INDEX(a.map_, a.i_, a.j_)].occ_dist > a.map_->cells[MAP_INDEX(b.map_, b.i_, b.j_)].occ_dist;
}

CachedDistanceMap*
get_distance_map(double scale, double max_dist)
{
  static CachedDistanceMap* cdm = NULL;

  if(!cdm || (cdm->scale_ != scale) || (cdm->max_dist_ != max_dist))
  {
    if(cdm)
      delete cdm;
    cdm = new CachedDistanceMap(scale, max_dist);
  }
  return cdm;
}
float scoreCell(double dist, double max_dist){
  double p1 = exp(-(dist * dist)/z_hit_denominator);
  return p1;
}

void enqueue(map_t* map, int i, int j,
	     int src_i, int src_j,
	     std::priority_queue<CellData>& Q,
	     CachedDistanceMap* cdm,
	     unsigned char* marked)
{
  if(marked[MAP_INDEX(map, i, j)])
    return;

  int di = abs(i - src_i);
  int dj = abs(j - src_j);
  double distance = cdm->distances_[di][dj];

  if(distance > cdm->cell_radius_)
    return;

  double dist = distance * map->scale;
  if(use_likelihood_model_exclude_unknown){
    //free区域才更新最短距离和评分
    if(map->cells[MAP_INDEX(map, i, j)].occ_state == -1){
      map->cells[MAP_INDEX(map, i, j)].occ_dist = dist;
      map->cells[MAP_INDEX(map, i, j)].score = scoreCell(dist, map->max_occ_dist);
    }
  }
  else{
    map->cells[MAP_INDEX(map, i, j)].occ_dist = dist;
    map->cells[MAP_INDEX(map, i, j)].score = scoreCell(dist, map->max_occ_dist);
  }

  CellData cell;
  cell.map_ = map;
  cell.i_ = i;
  cell.j_ = j;
  cell.src_i_ = src_i;
  cell.src_j_ = src_j;

  Q.push(cell);

  marked[MAP_INDEX(map, i, j)] = 1;
}

// Update the cspace distance values
void map_update_cspace(map_t *map, double max_occ_dist, double laser_standard_deviation)
{
  z_hit_denominator = 2*laser_standard_deviation*laser_standard_deviation;

  unsigned char* marked;
  std::priority_queue<CellData> Q;

  marked = new unsigned char[map->size_x*map->size_y];
  memset(marked, 0, sizeof(unsigned char) * map->size_x*map->size_y);
  map->max_occ_dist = max_occ_dist;

  CachedDistanceMap* cdm = get_distance_map(map->scale, map->max_occ_dist);

  // Enqueue all the obstacle cells
  CellData cell;
  cell.map_ = map;
  float lowest_score = scoreCell(max_occ_dist, map->max_occ_dist);
  map->min_score = lowest_score;
  for(int i=0; i<map->size_x; i++)
  {
    cell.src_i_ = cell.i_ = i;
    for(int j=0; j<map->size_y; j++)
    {
      if(map->cells[MAP_INDEX(map, i, j)].occ_state == +1)
      {
        map->cells[MAP_INDEX(map, i, j)].occ_dist = 0.0;
        map->cells[MAP_INDEX(map, i, j)].score = 1.0;
        cell.src_j_ = cell.j_ = j;
        marked[MAP_INDEX(map, i, j)] = 1;
        Q.push(cell);
      }
      else{
        map->cells[MAP_INDEX(map, i, j)].occ_dist = max_occ_dist;
        map->cells[MAP_INDEX(map, i, j)].score = lowest_score;
      }
    }
  }

  while(!Q.empty())
  {
    CellData current_cell = Q.top();
    if(current_cell.i_ > 0)
      enqueue(map, current_cell.i_-1, current_cell.j_, 
	      current_cell.src_i_, current_cell.src_j_,
	      Q, cdm, marked);
    if(current_cell.j_ > 0)
      enqueue(map, current_cell.i_, current_cell.j_-1, 
	      current_cell.src_i_, current_cell.src_j_,
	      Q, cdm, marked);
    if((int)current_cell.i_ < map->size_x - 1)
      enqueue(map, current_cell.i_+1, current_cell.j_, 
	      current_cell.src_i_, current_cell.src_j_,
	      Q, cdm, marked);
    if((int)current_cell.j_ < map->size_y - 1)
      enqueue(map, current_cell.i_, current_cell.j_+1, 
	      current_cell.src_i_, current_cell.src_j_,
	      Q, cdm, marked);

    Q.pop();
  }

  delete[] marked;
}

#ifdef __cplusplus
}
#endif

#endif