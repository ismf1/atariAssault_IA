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


using Index = std::tuple<size_t, size_t>;

// Devuelve dos índices diferentes
Index getIdxs(size_t s) {
    size_t rndixp, rndixm;
    do {
        rndixp = rand() % s; 
        rndixm = rand() % s; 
    } while(rndixp == rndixm);
    return { rndixp, rndixm };
}

void GA::breed(const VecIndividual &parents) {
    uint16_t notElite = popSize - elite;
    VecIndividual children;

    while (children.size() < notElite) {
        auto [ pidx, midx ] = getIdxs(parents.size()); 
        auto father = parents[pidx];
        auto mother = parents[midx];
        auto child  = father.crossover(mother, mutateRate);
        children.push_back(child);
    }

    population = parents;
    population.insert(population.end(), children.begin(), children.end());
}

Individual GA::evolve(uint16_t maxIterations, int16_t show) {
    for (size_t i = 0; i < maxIterations; i++) {
        std::cerr.setstate(std::ios_base::failbit);
        std::for_each(std::execution::par_unseq, population.begin(), population.end(), [](Individual &e) { e.fitness(); });
        std::cerr.clear();

        auto parents = selection();

        if (show != -1 && i % show == 0) {
            parents.front().fitness(true);
        }

        std::cout << "Best " << i << " fitness: " << parents.front().getFitness() << std::endl;
        std::cout << "Best " << i << " reward: " << parents.front().getReward() << std::endl;
        breed(parents);
    }

    return population.front();
}