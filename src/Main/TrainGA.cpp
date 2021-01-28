#include <GeneticAlgorithm/Individual.hpp>
#include <Utils/NSave.hpp>
#include <GeneticAlgorithm/GA.hpp>

using namespace std;

int main() {

    srand(time(NULL));
    NSave file("results/prueba_1010.000000.model");
    auto [ pesos, m ] = file.read(); 
    GA ga(2000, 0.05, 0.2, pesos);
    auto result = ga.evolve(34464, -1);

    cout << result.nn << endl;

    return 0;
}