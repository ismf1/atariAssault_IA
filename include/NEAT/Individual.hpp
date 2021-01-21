#pragma once

#include <NeuralNetworkv2/NeuralNet.hpp>
#include <NeuralNetworkv2/Types.hpp>

class Individual {

    private:
        NNet nn;

    public: 

    explicit Individual(const VecWeights &);
    void predict(const Mat2d &) const;
};  