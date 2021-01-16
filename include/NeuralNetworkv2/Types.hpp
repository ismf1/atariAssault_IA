#pragma once

#include <NeuralNetworkv2/Matrix.hpp>

using Vec2d      = Vector<double>;
using Mat2d      = Matrix<double>;
using ActFunc    = std::tuple<std::function<double (double)>, std::function<double (double)>>;
using CostFunc   = std::tuple<std::function<double (Mat2d, Mat2d)>, std::function<Mat2d (Mat2d, Mat2d)>>;
using VecActFunc = std::vector<ActFunc>;