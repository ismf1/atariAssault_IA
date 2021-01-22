#pragma once

#include <NEAT/Individual.hpp>

class GA
{
    using VecIndividual = std::vector<Individual>;

    private:
        uint16_t popSize;
        uint16_t elite;
        VecIndividual population;
        double mutateRate;

    public:
        explicit GA(uint16_t popSize, double mutate, double elite);
        auto selection();
        void breed(const VecIndividual &parents);
        Individual evolve(uint16_t maxIterations);
};