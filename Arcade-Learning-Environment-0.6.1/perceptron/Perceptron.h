#ifndef PERCEPTRON_H
#define PERCEPTRON_H
#include "Data.h"

class Perceptron{
    public:
        vector<float> w;             //Array de pesos actual
        vector<float> bestW;         //Mejor array de pesos
        int bestWCountInc;  //Aciertos del mejor array de pesos
        int inputParameters;      //Tamaño del array de pesos
        vector<int> faults;    //Indices de X fallidos
        vector<int> faultsNeg;
        vector<int> faultsPos;
        int nFaultsNeg; //Fallos en los que la salida real es negativa
        int nFaultsPos; //Fallos en los que la salida real es positiva
        float cotaFaults;
        float bestCotaFaults;

        Perceptron();
        void init(int params);
        void perceptronInit();
        void testEveryX(Data &data,int iPerceptron);
        void resetFaults();
        void saveIfBestWV();
        void updateWV(Data &data,int iPerceptron);
    
    private:
        int hw(vector<float> &X);
        float calcCotaFault(Data &data, int iPerceptron);
        void addFault(int iX,Data &data,int iPerceptron);
        bool testX(int iX,Data &data,int iPerceptron);
        void cpyWVectors(vector<float> &dest,vector<float> &source);
};

#endif
