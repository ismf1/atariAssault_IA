#include <vector>
#include <initializer_list>
#include <stdint.h>
#include <cstdio>
#include <stdexcept>
#include <random>
#include <queue>
#include <iostream>
#include <NeuralNetwork/NeuralNetwork.h>
#include <Utils/Data.hpp>
#include <Utils/Normalize.hpp>


using namespace std;

MatDouble_t vectorOfVectorsToMatDouble(const auto &vv){
    MatDouble_t m;

    for(size_t i=0;i<vv.size();i++){
        vector<double> vd;
        for(size_t j=0;j<vv[i].size();j++){
            vd.push_back(vv[i][j]);
        }
        m.push_back(vd);
    }

    return m;
}

void splitDataTrainTest(double percent,MatDouble_t const& X,MatDouble_t const& Y,
                        MatDouble_t& Xtrain,MatDouble_t& Ytrain,
                        MatDouble_t& Xval,MatDouble_t& Yval){
    size_t i=0;

    Xtrain.resize(0);
    Ytrain.resize(0);
    Xval.resize(0);
    Yval.resize(0);
    cout << "Separando train data..." << endl;
    while(i<(percent*X.size())){
        Xtrain.push_back(X[i]);
        Ytrain.push_back(Y[i]);
        i++;
    }
    cout << "Separando validation data..." << endl;
    while(i<X.size()){
        Xval.push_back(X[i]);
        Yval.push_back(Y[i]);
        i++;
    }
    cout << "Datos separados" << endl;
}

void run(){
    //Leemos los datos
    Data dataTrain;
    dataTrain.init("data/dataTrain.csv",59,5);
    Data dataVal;
    dataVal.init("data/dataVal.csv",59,5);

    //Escalamos los datos
    Normalize<double> scaler;
    dataTrain.X=scaler.fitTransform(Matrix(dataTrain.X)).toSTLVector();
    scaler.save("scaler.txt");

    dataVal.X=scaler.transform(Matrix(dataVal.X)).toSTLVector();

    cout << "Numero de datos:" << endl;
    for(size_t i=0;i<dataTrain.Yneg.size();i++){
        cout << "Neurona de salida " << i << endl;
        cout << "Negativos: " << dataTrain.Yneg[i] << endl;
        cout << "Positivos: " << dataTrain.Ypos[i] << endl;
        cout << "Totales: " << dataTrain.Y.size() << endl << endl;
    }


    //Generamos la red neuronal
    //Antes: initializer_list<uint16_t> layerStruct={dataTrain.tamXi,64,32,16,dataTrain.tamYi}; //Mejor resultado
    initializer_list<uint16_t> layerStruct={dataTrain.tamXi,64,32,16,dataTrain.tamYi};  
    float learningRate=0.03;
    uint16_t epochs=500;
    uint16_t patience=5;

    NeuralNetwork_t net(layerStruct,learningRate);
    cout << "Datos mínimos necesarios: ";
    uint16_t minData=0;
    for(size_t i=0;i<net.getm_layers().size();i++){
        minData+=net.getm_layers()[i].size()*net.getm_layers()[i][0].size();
    }
    minData=minData*10;
    cout << minData << endl << endl;

    //Entrenamos la red neuronal
    net.train(vectorOfVectorsToMatDouble(dataTrain.X),
              vectorOfVectorsToMatDouble(dataTrain.Y),
              vectorOfVectorsToMatDouble(dataVal.X),
              vectorOfVectorsToMatDouble(dataVal.Y),
              epochs,
              patience);
    cout << "Fallos:" << endl;
    MatDouble_t fallos=net.test(vectorOfVectorsToMatDouble(dataVal.X),vectorOfVectorsToMatDouble(dataVal.Y));
    double fallosSesgados=0;
    double fallosNoSesgados=0;

    for(size_t i=0;i<fallos.size();i++){
        double fallosSesgadosAux=0;
        double porcentajeFallosAux =(double)fallos[i][0]/dataVal.Yneg[i]*100; //Porcentaje de fallos negativos
        fallosSesgadosAux += porcentajeFallosAux * dataVal.Ypos[i] / dataVal.Y.size();   //Sesgo

        porcentajeFallosAux = (double)fallos[i][1]/dataVal.Ypos[i]*100; //Porcentaje de fallos positivos
        fallosSesgadosAux += porcentajeFallosAux * dataVal.Yneg[i] / dataVal.Y.size();   //Sesgo

        fallosSesgadosAux/=2;   //Media

        fallosSesgados+=fallosSesgadosAux;
        fallosNoSesgados+=(double)(fallos[i][1]+fallos[i][0])/dataVal.Y.size()*100;
    }

    fallosSesgados/=fallos.size();
    fallosNoSesgados/=fallos.size();

    cout << "Porcentaje total no sesgado de fallos: " << fallosNoSesgados << endl;
    cout << "Porcentaje total sesgado de fallos: " << fallosSesgados<< endl;

    net.save("NeuralNetwork.txt");
    
}

int main(){

    try{
        run();
    }catch(exception &e){
        printf("[EXCEPTION]: %s\n",e.what());
        return -1;
    }

    return 0;
}