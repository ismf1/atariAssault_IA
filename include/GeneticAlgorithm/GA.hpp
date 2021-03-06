#pragma once

#include <GeneticAlgorithm/Individual.hpp>

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
        explicit GA(uint16_t popSize, double mutateRate, double elite, const VecWeights &w);
        auto selection();
        void breed(const VecIndividual &parents);
        Individual evolve(uint16_t maxIterations, int16_t show = -1);
};