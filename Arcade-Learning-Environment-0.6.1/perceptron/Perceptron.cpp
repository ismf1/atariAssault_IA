#include "stdlib.h"
#include "stdio.h"
#include "stdbool.h"
#include "Perceptron.h"
#include "Data.h"
#include <iostream>

Perceptron::Perceptron(){
    
}
void Perceptron::init(int params){
    this->bestCotaFaults=99999;
    this->bestWCountInc=99999;
    this->inputParameters=params+1;

    this->w.resize(inputParameters);
    this->bestW.resize(inputParameters);

    //Inicializamos los pesos a 0
    for(int i=0;i<this->inputParameters;i++){
        this->w[i]=0;
        this->bestW[i]=0;
    }

    this->resetFaults();
}

//Prediccion del perceptron iPerceptron con la entrada X
int Perceptron:: hw(vector<float> &X){
    int sol=0;

    for(int i=0;i<this->inputParameters;i++){
        sol+=this->w[i]*X[i];
    }

    if(sol<0) return -1;
    else return 1;
}

void Perceptron:: resetFaults(){
    this->nFaultsNeg=0;
    this->nFaultsPos=0;
    this->cotaFaults=999999;
    this->faults.resize(0);
    this->faultsNeg.resize(0);
    this->faultsPos.resize(0);
}

float Perceptron:: calcCotaFault(Data &data,int iPerceptron){

    float cotaPos=(float)this->nFaultsPos/data.Ypos[iPerceptron];   //Sin comprobar
    //Si no hay datos positivos, cada fallo sera muy grave
    if(data.Ypos[iPerceptron]==0){
        cotaPos=(float)this->nFaultsPos/0.001;
    }
    float cotaNeg=(float)this->nFaultsNeg/data.Yneg[iPerceptron];   //Sin comprobar
    //Si no hay datos negativos, cada fallo sera muy grave
    if(data.Yneg[iPerceptron]==0){
        cotaNeg=(float)this->nFaultsNeg/0.001;
    }

    float cota=cotaPos+cotaNeg/2;

    return cota;
}

void Perceptron:: addFault(int iX,Data &data,int iPerceptron){
    this->faults.push_back(iX);
 
    //Actualizamos valores nFaultsNeg y nFaultsPos
    if(data.Y[iX][iPerceptron]<0){
        this->nFaultsNeg++;
        this->faultsNeg.push_back(iX);
    }
    else {
        this->nFaultsPos++;
        this->faultsPos.push_back(iX);
    }
}

//Testea un dato de entrada con el perceptron X
bool Perceptron:: testX(int iX,Data &data,int iPerceptron){
    if(this->hw(data.X[iX]) == data.Y[iX][iPerceptron]){
        return true;
    }else{
        return false;
    }
}

//Testea todos los datos de entrada con el perceptron X
void Perceptron:: testEveryX(Data &data,int iPerceptron){
    for(int i=0;i<data.X.size();i++){
        //Si no acierta la prediccion
        if(!this->testX(i,data,iPerceptron)){
            addFault(i,data,iPerceptron);
        }
    }
    this->cotaFaults = calcCotaFault(data,iPerceptron);
    //if(this->cotaFaults>0)std::cout << this->cotaFaults << std::endl; //BORRAR
}

void Perceptron:: cpyWVectors(vector<float> &dest,vector<float> &source){
    for(int i=0;i<source.size();i++){
        dest[i]=source[i];
    }
}

void Perceptron:: saveIfBestWV(){
    //Si mejoran los aciertos
    if(this->cotaFaults < this->bestCotaFaults){
        //Actualizar bestCotaFaults
        this->bestCotaFaults=this->cotaFaults;
        //Actualizar bestWCountInc
        this->bestWCountInc=this->faults.size();
        //Actualizar bestW
        cpyWVectors(this->bestW,this->w);
    }
}

void Perceptron:: updateWV(Data &data, int iPerceptron){
    int negPosRand=rand()%2;

    if(this->nFaultsNeg==0) negPosRand=1;
    if(this->nFaultsPos==0) negPosRand=0;

    int iFaultX;
    int iX;
    if(negPosRand==0 && this->nFaultsNeg!=0){
        //Actualizar con X cuya Y real es negativa
        iFaultX=rand()%this->nFaultsNeg;  //Escogemos indice de X mal clasificado aleatoriamente
        iX=this->faultsNeg[iFaultX]; //Seleccionamos el indice de X
    }else if(this->nFaultsPos!=0){
        //Actualizar con X cuya Y real es positiva
        iFaultX=rand()%this->nFaultsPos;  //Escogemos indice de X mal clasificado aleatoriamente
        iX=this->faultsPos[iFaultX]; //Seleccionamos el indice de X
    }else{
        return;
    }

    for(int i=0;i<this->inputParameters;i++){
        //w=w+xr*yr
        this->w[i]+=data.X[iX][i]*data.Y[iX][iPerceptron];
    }

}
