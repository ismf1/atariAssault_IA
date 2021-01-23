#include <iostream>
#include <Utils/Data.hpp>
#include <Utils/Normalize.hpp>
#include <NeuralNetworkv2/Types.hpp>
#include <NeuralNetworkv2/Functions.hpp>
#include <NEAT/Individual.hpp>
#include <cassert>
#include <omp.h>

using namespace std;

void calculateTime(auto &&l, string des) {
    double itime, ftime, exec_time;
    itime = omp_get_wtime();
    l();
    ftime = omp_get_wtime();
    exec_time = ftime - itime;
    printf(des.c_str(), exec_time);
}

int main(int argc, char *argv[]) {

    Individual i, i2; 

    auto x = i.crossover(i2, 0.1);

    assert(i.nn.getWeights() != x.nn.getWeights());
    assert(i2.nn.getWeights() != x.nn.getWeights());

    std::cout << i.nn << std::endl;
    std::cout << i2.nn << std::endl;
    std::cout << x.nn << std::endl;

    // Data data;

    // data.init(argv[1], 59, 5);

    // auto x = data.X;
    // auto y = data.Y;

    // Matrix<float> mat(x);

    // cout << scaler.fitTransform(mat) << endl;

    // Mat2d m(748790*2, 59, []() { return Functions::rand(0, 1000); });
    // Mat2d m2(59, 128, []() { return Functions::rand(0, 1000); });

    // calculateTime([&m, &m2]() { m.parallelDot(m2); }, "Time parallel taken: %f\n");
    // calculateTime([&m, &m2]() { m * m2; }, "Time not parallel taken: %f\n");


    // tStart = clock();
    // printf("Time parallel taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);

    // cout << "Transformación: " << endl;
    // cout << scaler.fitTransform(m) << endl;

    // cout << "Valores Minimos - Maximos por columna: " << endl;
    // for (auto const &e : scaler.getMinMax()) {
    //     auto [min, max] = e;
    //     cout << "(" << min << ", " << max << ")" << endl; 
    // }
    // cout << "Ejemplo de transformación:" << endl;
    // cout << scaler.transform(Vec2d({ 1, 3, 4, 6 })) << endl;
    // cout << "Ejemplo de escritura:" << endl;
    // scaler.save("prueba.txt");
    // cout << "Ejemplo de lectura:" << endl;
    // Normalize<double> ejemplo;
    // ejemplo.load("prueba.txt");
    // for (auto const &e : ejemplo.getMinMax()) {
    //     auto [min, max] = e;
    //     cout << "(" << min << ", " << max << ")" << endl; 
    // }
    // cout << "Ejemplo pasar una matrix a una matrix normal" << endl;
    // m.toSTLVector();
    // cout << "Ejemplo pasar un vector a un vector normal" << endl;
    // Vec2d vec(m[0]);
    // vec.toSTLVector();

    return 0;
}