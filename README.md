# BaB-Global-Localization
Efficient  global localization method using branch and bound based on occupancy grid map and 2D-LiDAR

System: Ubuntu 20.04

Require: c++14  openmp  opencv  YAMLCPP

Compile
-------
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make

Use: run "bab_global_localization" by specifying the dataset path.
-------
    ./bab_global_localization intel/

Video
-------
https://github.com/DuSongGit/BaB-Global-Localization/issues/1#issue-2754177207
