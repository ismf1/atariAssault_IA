#include "NSave.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>


void NSave::escribir(const std::vector< std::vector< std::vector<double> > > v,const std::string s) const{

    std::ofstream ficheroSalida;
    ficheroSalida.open (s);
    ficheroSalida << "[";
    for(int i=0;i<v.size();i++){
        ficheroSalida << "[";
        for(int j=0;j<v[i].size();j++){
            ficheroSalida << "[";
            for(int k=0;k<v[i][j].size();k++){
                ficheroSalida << v[i][j][k];

                if(v[i][j].size()-1>k)
                    ficheroSalida <<",";
            }
            ficheroSalida << "]";
        }
        ficheroSalida << "]";
    }
    ficheroSalida << "]";
    
    ficheroSalida.close();

}

std::vector< std::vector< std::vector<double> > > NSave::leer(const std::string fichero) const{
    std::vector<  std::vector<  std::vector<double> > > x;
    std::vector<  std::vector<double> > y;
    std::vector<double> z;
    std::string s;
    s="";

    std::ifstream ficheroEntrada;
    char letra;

    ficheroEntrada.open (fichero);
    ficheroEntrada >> letra;
    while (! ficheroEntrada.eof() ) {
        ficheroEntrada >> letra;
        while(letra!=']'){
            while(letra!=']'){
                ficheroEntrada >> letra;
                if(letra=='['){
                    ficheroEntrada >> letra;
                }
                if(letra==']' || letra==','){
                    std::cout << s << std::endl;
                    z.push_back(atof(s.c_str()));
                    s="";
                }else{
                    s+=letra;
                }
                
            }
            y.push_back(z);
            z.clear();
            ficheroEntrada >> letra;
        }
        if(y.size()>0){
            x.push_back(y);
            y.clear();
        }
    }
    ficheroEntrada.close();
    escribir(x,"leida");

    return x;
}