# BaB-Global-Localization
Efficient  Global Localization Method Using Branch-and-Bound

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