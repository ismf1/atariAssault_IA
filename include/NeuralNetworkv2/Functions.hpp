#pragma once

#include <cmath>
#include <random>
#include <functional>
#include <NeuralNetworkv2/Vector.hpp>
#include <NeuralNetworkv2/Matrix.hpp>
#include <NeuralNetworkv2/Types.hpp>

class Functions {

    public: 

    static double rand(double min, double max){
        static std::random_device dev;  //Coge numero de dispositivo hardware preparado para generacion de aleatorios
        static std::mt19937 rng(dev()); //Algoritmo pseudoaleatorio
        static std::uniform_real_distribution<double> dist(min,max);    //Distribucion lineal

        return dist(rng);
    }

    static double relu(double x) {
        return x > 0? x : 0;
    }

    static double reluD(double x) {
        return x <= 0? 0 : 1;
    }

    static double sigm(double x) {
        return 1.f / (1.f + std::pow(std::exp(1.f), -x));
    }

    static double sigmD(double x) {
        return x * (1 - x);
    }

    static double mse(Mat2d yp, Mat2d yr) {
        return (yp - yr).pow(2).mean();
    }

    static Mat2d mseD(Mat2d yp, Mat2d yr) {
        return yp - yr;
    }
};
