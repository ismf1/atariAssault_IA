#include <vector>
#include <stdint.h>
#include <cstdio>
#include <stdexcept>
#include <iostream>

using namespace std;

void multiplyIntVectors(auto n, auto &v){ //FUNCIONA
    for(size_t i=0;i<v.size();i++){
        v[i]=v[i]*n;
    }
}

void subVectors(auto &v1,auto &v2){ //FUNCIONA
    if(v1.size()!=v2.size()){
        throw length_error("Vectors must have the same size when sub.");
    }

    for(size_t i=0;i<v1.size();i++){
        v1[i]=v1[i]-v2[i];
    }
}

void multiplyIntMatrix(auto n, auto &v){   //Funciona
    for(size_t i=0;i<v.size();i++){
        for(size_t j=0;j<v[i].size();j++){
            for(size_t k=0;k<v[i][j].size();k++){
                v[i][j][k]=v[i][j][k]*n;
            }
        }
    }
}

void subMatrix(auto &v1,auto &v2){   //Funciona
    if(v1.size()!=v2.size()){
        throw length_error("Vectors must have the same size when sub.");
    }

    for(size_t i=0;i<v1.size();i++){
        for(size_t j=0;j<v1[i].size();j++){
            for(size_t k=0;k<v1[i][j].size();k++){
                v1[i][j][k]=v1[i][j][k]-v2[i][j][k];
            }
        }
    }
}