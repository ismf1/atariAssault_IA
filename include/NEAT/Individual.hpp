#pragma once

#include <NeuralNetworkv2/NeuralNet.hpp>
#include <NeuralNetworkv2/Types.hpp>
#include <ale_interface.hpp>

class Individual {

    private:
        NNet nn;
        double fit;

    public: 

    explicit Individual();
    explicit Individual(const VecWeights &);
    Individual(const Individual &);
    double fitness();
    Individual crossover(const Individual &, double mutateRate);
    double agentStep(ALEInterface &alei);
    const std::vector<double> getState(ALEInterface &alei);
    inline double getFitness() const { return fit; }
    bool operator<(const Individual& other) const;
};  