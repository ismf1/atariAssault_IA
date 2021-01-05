#ifndef DATA_H
#define DATA_H
#include <stdio.h>
#include <vector>
using namespace std;

class Data{
    public:
        int tamXi = 7;
        int tamYi = 2;
        static const int MAXCHAR = 1000;
        const char* filename;
        int size;
        vector<vector<float>> X;
        vector<vector<int>> Y;
        vector<int> Yneg;
        vector<int> Ypos;

        Data();
        int countLinesFile();
        void init(const char* filename,const int tamXi,const int tamYi);
        void splitData(char str[],int line);
};

#endif
