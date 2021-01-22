#include <NEAT/Individual.hpp>
#include <NEAT/GA.hpp>

using namespace std;

int main() {
    GA ga(10, 0.1, 0.3);
    auto result = ga.evolve(100);

    cout << result.getFitness() << endl;

    return 0;
}