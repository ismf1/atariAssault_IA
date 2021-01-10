#pragma once

#include <stdint.h>
#include <Types.hpp>

struct NeuralLayer {

    using Weights = std::vector<std::vector<double>>;

    int16_t nconn;
    int16_t nneur;
    ActFunc actf;
    Mat2d w;
    Vec2d b;

    explicit NeuralLayer(int16_t nconn, int16_t nneur, ActFunc actf);
    explicit NeuralLayer(Weights w);
    Weights getWeights() const;

    friend std::ostream& operator<<(std::ostream &os, const NeuralLayer &nl);
};