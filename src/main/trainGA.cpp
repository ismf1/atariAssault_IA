#include <NEAT/Individual.hpp>
#include <NEAT/GA.hpp>

using namespace std;

int main() {
    GA ga(15, 0.2, 0.3);
    auto result = ga.evolve(500, 10);

    cout << result.nn << endl;

    return 0;
}