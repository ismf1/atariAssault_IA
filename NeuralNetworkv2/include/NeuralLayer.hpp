#pragma once

#include <stdint.h>
#include <Types.hpp>

struct NeuralLayer {
    int16_t nconn;
    int16_t nneur;
    ActFunc actf;
    Mat2d w;
    Vec2d b;

    explicit NeuralLayer(int16_t nconn, int16_t nneur, ActFunc actf);

    std::ostream& toStream(std::ostream&) const;
    friend std::ostream& operator<<(std::ostream &os, const NeuralLayer &nl);
};