#include <NSave.hpp>
#include <iostream>
#include <fstream>

void NSave::write(const NSave::Data &v) const {

    std::ofstream fileOut;
    fileOut.open (fileName);
    fileOut << "[";
    for(size_t i = 0; i < v.size();i++){
        fileOut << "[";
        for(size_t j = 0; j < v[i].size();j++){
            fileOut << "[";
            for(size_t k = 0; k < v[i][j].size();k++){
                fileOut << v[i][j][k];

                if(v[i][j].size()-1>k)
                    fileOut <<" ";
            }
            fileOut << "]";
        }
        fileOut << "]";
    }
    fileOut << "]";
    
    fileOut.close();

}

NSave::Data NSave::read() const {
    Data x;
    std::vector<std::vector<double>> y;
    std::vector<double> z;
    std::string s;
    std::ifstream fileEntrada;
    char letra;
    
    fileEntrada.open (fileName);
    fileEntrada >> letra;

    while (! fileEntrada.eof() ) {
        fileEntrada >> letra;
        while(letra!=']'){
            while(letra!=']'){
                fileEntrada >> letra;
                if(letra!=' ' && letra!=']' && letra!='['){
                    s=letra;
                    z.push_back(atof(s.c_str()));
                }
            }
            y.push_back(z);
            z.clear();
            fileEntrada >> letra;
        }
        if(y.size()>0){
            x.push_back(y);
            y.clear();
        }
    }
    fileEntrada.close();

    return x;
}