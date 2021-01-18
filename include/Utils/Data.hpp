#ifndef DATA_H
#define DATA_H
#include <stdio.h>
#include <vector>
#include <stdint.h>
using namespace std;

class Data{
    public:
        uint16_t tamXi = 7;
        uint16_t tamYi = 2;
        static const int MAXCHAR = 1000;
        const char* filename;
        int size;
        vector<vector<double>> X;
        vector<vector<double>> Y;
        vector<double> Yneg;
        vector<double> Ypos;

        Data();
        int countLinesFile();
        void init(const char* filename,const int tamXi,const int tamYi);
        void splitData(char str[],int line);
};

#endif
