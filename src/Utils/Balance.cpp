#include <Utils/Balance.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cstdlib>
#include <tuple>
#include <cmath>
#include <math.h>
#include <ctime>
#include <iostream>

//inicializamos el numero de casos de cada clase
Balance::Balance(const Vec_st &v){
    col1=v[0];
    col2=v[1];
    col3=v[2];
    col4=v[3];
    col5=v[4];
    srand(time(NULL));
}

//generamos el nuevo ejemplo con la muestra y el vecino
Vec_d Balance::calculaVecino(Vec_d const& vecino, Vec_d const& original){

    Vec_d nuevo;
    
    for(size_t i=0;i<original.size();i++){
        double newValor=0;
        newValor=abs(original[i]-vecino[i]);
        double porce=(rand()%(100+1));
        newValor=(newValor*(porce/100))+original[i];
        int ent=newValor;
        nuevo.push_back(ent);
    }

    return nuevo;
}

//devuelve el vecino de forma aleatoria
Vec_d Balance::generateVecino(Mat_d const& m, size_t const& i){


    int vecino=rand()%(4)+1;

    switch(vecino){
        case 1:
            if((i-vecino)>0){
                vecino=i-2;
                return m[vecino];
            }else{
                vecino=i+1;
                return m[vecino];
            }
            break;
        case 2:
            if((i-vecino)>0){
                vecino=i-1;
                return m[vecino];
            }else{
                vecino=i+1;
                return m[vecino];
            }
            break;
        case 3:
            if((i+vecino)<m.size()){
                vecino=i+1;
                return m[vecino];
            }else{
                vecino=i-1;
                return m[vecino];
            }
            break;
        case 4:
            if((i+vecino)<m.size()){
                vecino=i+2;
                return m[vecino];
            }else{
                vecino=i-1;
                return m[vecino];
            }
            break;

    }
    

    return m[vecino];
}

//generamos el los nuevos ejemplos
std::tuple<Mat_d, Mat_d> Balance::generate(Mat_d const& X, Mat_d const& y){

    int muestras1=col5-col1;
    int muestras2=col5-col2;
    int muestras3=col5-col3;
    Mat_d newX;
    Mat_d newY;

    for(size_t i=0;i<y.size();i++){
        newX.push_back(X[i]);
        newY.push_back(y[i]);
        if(y[i][0]==1 && muestras1>0){
            muestras1--;
            col1++;
            Vec_d v=generateVecino(X,i);
            Vec_d v1=calculaVecino(v,X[i]);
            newX.push_back(v1);
            newY.push_back(y[i]);
        }else if(y[i][1]==1 && muestras2>0){
            col2++;
            muestras2--;
            Vec_d v=generateVecino(X,i);
            Vec_d v1=calculaVecino(v,X[i]);
            newX.push_back(v1);
            newY.push_back(y[i]);
        }else if(y[i][2]==1 && muestras3>0){
            col3++;
            muestras3--;
            Vec_d v=generateVecino(X,i);
            Vec_d v1=calculaVecino(v,X[i]);
            newX.push_back(v1);
            newY.push_back(y[i]);
        }
        
    }

    return {newX,newY};
    
}



