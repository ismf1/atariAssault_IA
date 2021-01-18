#include <iostream>
#include <Utils/Data.hpp>
#include <Utils/Normalize.hpp>
#include <NeuralNetworkv2/Types.hpp>

using namespace std;

int main(int argc, char *argv[]) {

    // Data data;

    // data.init(argv[1], 59, 5);

    // auto x = data.X;
    // auto y = data.Y;

    // Matrix<float> mat(x);

    // cout << scaler.fitTransform(mat) << endl;

    Normalize<double> scaler;
    Mat2d m({
        { 0, 2, 3, 40},
        { 0, 10, 30, 4},
        { 0, 10, 30, 4},
        { 0, 10, 30, 4},
        { 0, 2, 5, 4},
        { 0, 2, 3, 4},
    });
    cout << "Transformación: " << endl;
    cout << scaler.fitTransform(m) << endl;

    cout << "Valores Minimos - Maximos por columna: " << endl;
    for (auto const &e : scaler.getMinMax()) {
        auto [min, max] = e;
        cout << "(" << min << ", " << max << ")" << endl; 
    }
    cout << "Ejemplo de transformación:" << endl;
    cout << scaler.transform(Vec2d({ 1, 3, 4, 6 })) << endl;
    cout << "Ejemplo de escritura:" << endl;
    scaler.save("prueba.txt");
    cout << "Ejemplo de lectura:" << endl;
    Normalize<double> ejemplo;
    ejemplo.load("prueba.txt");
    for (auto const &e : ejemplo.getMinMax()) {
        auto [min, max] = e;
        cout << "(" << min << ", " << max << ")" << endl; 
    }
    cout << "Ejemplo pasar una matrix a una matrix normal" << endl;
    m.toSTLVector();
    cout << "Ejemplo pasar un vector a un vector normal" << endl;
    Vec2d vec(m[0]);
    vec.toSTLVector();

    return 0;
}