#include "NSave.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <tuple>


void NSave::write(const Data &v, const Func &t) const{

    std::ofstream outFile;
    outFile.open (fileName);

    outFile << "(";

    for(size_t i=0;i<t.size();i++){
        outFile << t[i];
        if(t.size()-1>i)
            outFile <<",";
    }
    outFile << ")";
    outFile << "[";
    for(size_t i=0;i<v.size();i++){
        outFile << "[";
        for(size_t j=0;j<v[i].size();j++){
            outFile << "[";
            for(size_t k=0;k<v[i][j].size();k++){
                outFile << v[i][j][k];

                if(v[i][j].size()-1>k)
                    outFile <<",";
            }
            outFile << "]";
        }
        outFile << "]";
    }
    outFile << "]";

    

    outFile.close();

}

std::tuple<NSave::Data,NSave::Func> NSave::read() const{
    Data x;
    std::vector<  std::vector<double> > y;
    std::vector<double> z;
    std::string s;
    s="";
    NSave::Func t; 

    std::ifstream fileIn;
    char letra;

    fileIn.open (fileName);
    fileIn >> letra;
    if(letra=='('){
        std::string fun="";
        while(letra!=')'){
            fileIn >> letra;
            if(letra==',' || letra==')'){
                switch(stoi(fun)){
                    case 0:
                        t.push_back(SIGM);
                        break;
                    case 1:
                        t.push_back(RELU);
                        break;
                }
                fun="";
            }else{
                fun+=letra;
            }
        }
    }
    fileIn >> letra;
    while (! fileIn.eof() ) {
        fileIn >> letra;
        while(letra!=']' && letra!='('){
            while(letra!=']'){
                fileIn >> letra;
                if(letra=='['){
                    fileIn >> letra;
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
            fileIn >> letra;
        }
        if(y.size()>0){
            x.push_back(y);
            y.clear();
        }
    }
    
    fileIn.close();

    std::tuple<NSave::Data,NSave::Func> tupla (std::make_tuple(x,t));
    
    return tupla;
}