#pragma once

// #include <stdarg.h>

// #include <cuda.h>


#include "torch/torch.h"

#include <Eigen/Core>



class Sphere{
public:
    Sphere(const float radius, const Eigen::Vector3f center);
    ~Sphere();

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>  ray_intersection(const torch::Tensor& ray_origins, const torch::Tensor& ray_dirs);
    torch::Tensor rand_points_inside(const int nr_points);


    float m_radius;
    Eigen::Vector3f m_center;
    torch::Tensor m_center_tensor;

private:
    

  
};
