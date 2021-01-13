#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <initializer_list>
#include <stdint.h>
#include <cstdio>
#include <stdexcept>
#include <random>
#include <queue>
#include <iostream>
#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include <Network.hpp>
using namespace std;

using VecDouble_t = vector<double>; //Neurona
/* W    [ w0(........) ]
        [ w1(........) ]
        [ ............ ]
        [ wn(........) ]*/
using MatDouble_t = vector<VecDouble_t>;    //Capa
enum class ActF{SIGMOID,RELU};

struct NeuralNetwork_t : public Network {
    explicit NeuralNetwork_t() ;
    explicit NeuralNetwork_t(initializer_list<uint16_t> const& layers,float learningR) ;

    void setActiveFunctions(initializer_list<ActF> v);

    VecDouble_t multiplyT(VecDouble_t const& input,MatDouble_t const& W) const;

    double activeFunction(auto x, auto layer) const;

    constexpr double sigmoid(auto x) const;
    constexpr double relu(double x) const;

    double activeFunctionDeriv(auto x, auto layer) const;

    constexpr auto sigmoidDeriv(auto x) const;
    constexpr auto reluDeriv(auto x) const;

    //constexpr auto signal(VecDouble_t const& x,auto layer,auto neuron);

    constexpr auto deltaOutputLayer(VecDouble_t const& y,auto layer,auto neuron);

    constexpr auto deltaHiddenLayers(auto layer,auto neuron);

    auto delta(VecDouble_t const& y,auto layer,auto neuron);

    //Cambiar para y multidimensional
    constexpr auto errorDerivateFunction(auto hx, auto y,auto size_y) const;

    //Cambiar para y multidimensional
    constexpr auto errorFunctionInNeuron(double hx, double y) const;

    auto errorFunction(const VecDouble_t hx,const VecDouble_t y) const;

    //Cambiar para y multidimensional
    double errorFunctionVector(MatDouble_t const& X, MatDouble_t const& Y);

    //Derivada parcial para el peso m_layers[layer][neuron][beforeNeuron]
    double errorDerivateParcialFunction(VecDouble_t const& x,VecDouble_t const& y,auto layer,auto neuron,auto beforeNeuron);
    //Devuelve vector de las derivadas parciales de la funcion de error de 1 neurona
    VecDouble_t errorDerivateParcialFunctions(VecDouble_t const& x,VecDouble_t const& y,auto layer,auto neuron);

    //Actualiza los pesos de una capa
    MatDouble_t updateLayer(VecDouble_t const& x,VecDouble_t const& y,auto layer);

    //Actualiza los pesos de la red
    void updateWeights(VecDouble_t const& x,VecDouble_t const& y);

    void calculateFeedForwardMat(VecDouble_t const& x);

    VecDouble_t feedforwardinlayer(VecDouble_t const& x,auto layer);

    double feedforwardinneuron(VecDouble_t const& x,auto layer,auto neuron);

    void train(MatDouble_t const& X,MatDouble_t const& Y,MatDouble_t const& Xval,MatDouble_t const& Yval,uint16_t epochs, uint16_t patience);

    VecDouble_t activeFunction (VecDouble_t const& vec, auto layer);

    VecDouble_t relu (VecDouble_t const& vec) const;

    VecDouble_t sigmoid (VecDouble_t const& vec) const;

    VecDouble_t feedforward(VecDouble_t const& x);

    VecDouble_t predict(VecDouble_t const& x);

    MatDouble_t createCopyLayer(MatDouble_t const& layer);

    vector<MatDouble_t> getm_layers();

    double evaluateNet(MatDouble_t const& X, VecDouble_t const& Y);

    void load(const string &fichero);

    void save(const string s) const;

    MatDouble_t test(MatDouble_t const& X, MatDouble_t const& Y);

private:
    vector<ActF> functionsAct;
    MatDouble_t feedforwardMat;
    MatDouble_t signalMat;
    vector<MatDouble_t> m_layers;
    queue<VecDouble_t> deltaQueue;
    float learningRate=0.2;
    //Para error sesgado
    vector<int> Yneg;
    vector<int> Ypos;
};

#endif