#include <NEAT/GA.hpp>
#include <NeuralNetworkv2/Functions.hpp>
#include <vector>
#include <execution>
#include <parallel/algorithm>

GA::GA(uint16_t popSize, double mutateRate, double elite)
    : popSize(popSize), mutateRate(mutateRate), elite(elite * popSize), population(popSize)
{
    std::generate(population.begin(), population.end(), []() { return Individual(); });
}

auto GA::selection() {
    std::sort(population.begin(), population.end());
    return std::vector(population.begin(), population.begin() + elite);
}

void GA::breed(const VecIndividual &parents) {
    uint16_t notElite  = popSize - elite;
    VecIndividual children;

    while (children.size() < notElite) {
        auto father = Functions::randomChoice(parents);
        auto mother = Functions::randomChoice(parents);
        auto child  = father.crossover(mother, mutateRate);
        children.push_back(child);
    }

    population = parents;
    population.insert(population.end(), children.begin(), children.end());
}

Individual GA::evolve(uint16_t maxIterations) {
    for (size_t i = 0; i < maxIterations; i++) {
        std::for_each(std::execution::par_unseq, population.begin(), population.end(), [](Individual &e) { e.fitness(); });
        auto parents = selection();
        std::cout << "Best " << i << "fitness: " << parents.front().getFitness() << std::endl;
        breed(parents);
    }
    
    return population.front();
}