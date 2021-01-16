#include <Utils/NSave.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>


void NSave::write(const Data &v) const{

    std::ofstream outFile;
    outFile.open (fileName);
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

NSave::Data NSave::read() const{
    Data x;
    std::vector<  std::vector<double> > y;
    std::vector<double> z;
    std::string s;
    s="";

    std::ifstream fileIn;
    char letra;

    fileIn.open (fileName);
    fileIn >> letra;
    while (! fileIn.eof() ) {
        fileIn >> letra;
        while(letra!=']'){
            while(letra!=']'){
                fileIn >> letra;
                if(letra=='['){
                    fileIn >> letra;
                }
                if(letra==']' || letra==','){
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

    return x;
}