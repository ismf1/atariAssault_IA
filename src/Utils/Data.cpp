#include <Utils/Data.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

Data::Data(){

}

int Data::countLinesFile(){
    int n=0;

    FILE *fp;
    char str[Data::MAXCHAR];
 
    fp = fopen(this->filename, "r");
    if (fp == NULL){
        printf("Could not open file %s",this->filename);
        return 1;
    }
    while (fgets(str, Data::MAXCHAR, fp) != NULL)
        //printf("%s", str);
        n++;
    fclose(fp);

    return n;
}

void Data::init(const char* filename,const int tamXi,const int tamYi){
    this->filename=filename;
    this->tamXi=tamXi;
    this->tamYi=tamYi;

    this->size=countLinesFile();
    this->X.resize(this->size,vector<double>(tamXi));
    this->Y.resize(this->size,vector<double>(tamYi));
    this->Yneg.resize(tamYi);
    this->Ypos.resize(tamYi);

    for(int i=0;i<tamYi;i++){
        this->Yneg[i]=0;
        this->Ypos[i]=0;
    }

    FILE *fp;
    char str[MAXCHAR];
 
    fp = fopen(filename, "r");
    if (fp == NULL){
        printf("Could not open file %s",filename);
        //FALTA: Poner throw exception
    }
    int line=0;
    while (fgets(str, MAXCHAR, fp) != NULL){
        std::cout << "Loading line " << line << "..." << std::endl;
        splitData(str,line);
        line++;
    }
    fclose(fp);
}

void Data:: splitData(char str[],int line){

    int i=0;
    float n=-1;
    int ny=-1;

    //this->X[line][0]=1;  //El valor 0 de X multiplica al BIAS

    //Decodificamos X
        //Decodificar TamXi(7) valores
    for(int j=0;j<this->tamXi;j++){
            //Decodificamos valor j
        char strAux[15];
        int nAux=0;
        while(str[i]!=';'){
            strAux[nAux]=str[i];
            nAux++;
            i++;
        }
        i++;
        strAux[nAux]='\0';

        n=atof(strAux);
            //Guardamos valor
        this->X[line][j]=n;

    }

    //Decodificamos Y
        //Decodificar TamYi(7) valores
    for(int j=0;j<this->tamYi;j++){
            //Decodificamos valor j
        char strAux[15];
        int nAux=0;
        while(str[i]!=';' && str[i]!='\n'){
            strAux[nAux]=str[i];
            nAux++;
            i++;
        }i++;
        strAux[nAux]='\0';

        ny=atoi(strAux);
        if(ny==0){
            //ny=-1;  //Convertimos los 0 a -1
            this->Yneg[j]++;   ////Sin comprobar
        }else{
            this->Ypos[j]++;   ////Sin comprobar
        }
            //Guardamos valor
        this->Y[line][j]=ny;


    }
    
}
