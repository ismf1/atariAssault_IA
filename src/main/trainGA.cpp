#include <NEAT/Individual.hpp>
#include <NEAT/GA.hpp>

using namespace std;

int main() {
    srand(time(NULL));
    GA ga(15, 0.05, 0.2);
    auto result = ga.evolve(500, 10);

    cout << result.nn << endl;

    return 0;
}