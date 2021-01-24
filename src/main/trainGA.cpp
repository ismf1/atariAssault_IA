#include <NEAT/Individual.hpp>
#include <Utils/NSave.hpp>
#include <NEAT/GA.hpp>

using namespace std;

int main() {

    srand(time(NULL));
    NSave file("results/prueba_1010.000000.model");
    auto [ pesos, mierda ] = file.read(); 
    GA ga(2000, 0.05, 0.2, pesos);
    auto result = ga.evolve(100000, 5);

    cout << result.nn << endl;

    return 0;
}