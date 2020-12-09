#include "stdlib.h"
#include "stdio.h"
#include <iostream>
using namespace std;
#include <vector>
#include <string>
#include <time.h>
#include "Data.h"
#include "Perceptron.h"
#include <fstream>
const int MAX_ITER = 1000;
const char* FILENAME="../data.txt";
const int TAM_XI = 59;
//const int TAM_XI = 128;
const int TAM_YI = 5;
const int N_PERCEPTRONS = TAM_YI;

Data data;
Perceptron perceptron[N_PERCEPTRONS];

//Inicializar todos los perceptrones
void perceptronsInit(){
    for(int i=0;i<N_PERCEPTRONS;i++){
        perceptron[i].init(TAM_XI);
    }
}

void init(){
    srand(time(NULL)); // Intializes random number generator
    perceptronsInit();
    data.init(FILENAME,TAM_XI,TAM_YI);
}

void saveResult(){
    std::ofstream outfile;

    outfile.open("IA.txt", std::ios_base::app);

    outfile << "{";
    for(int i=0;i<N_PERCEPTRONS;i++){
        outfile << "{";
        for(int j=0;j<perceptron[i].inputParameters;j++){
            outfile << perceptron[i].bestW[j] << ',';
        }
        outfile << "}";
        if(i!=N_PERCEPTRONS-1) outfile << ",";
    }
    outfile << "}\n";
}

void printResult(){
    printf("\n");
    for(int i=0;i<N_PERCEPTRONS;i++){
        printf("Perceptron %i\n",i);
        printf("   Vector de pesos: \n");
        for(int j=0;j<perceptron[i].inputParameters;j++){
            printf("%f ",perceptron[i].bestW[j]);
        }
        printf("\n");

        printf("   Cota fallos: %f\n",perceptron[i].bestCotaFaults);
    }

    printf("\n");
    printf("{");
    for(int i=0;i<N_PERCEPTRONS;i++){
        printf("{");
        for(int j=0;j<perceptron[i].inputParameters;j++){
            printf("%f, ",perceptron[i].bestW[j]);
        }
        printf("}");
        if(i!=N_PERCEPTRONS-1) printf(",");
    }
    printf("}\n");
}

int main(){
    init();
    for(int i=0;i<N_PERCEPTRONS;i++){
        //Perceptron i
        cout << "Perceptron " << i << endl;
        //----------BORRAR-----------------
        cout << "Negativos: " << data.Yneg[i] << endl;
        cout << "Positivos: " << data.Ypos[i] << endl;
        cout << "Totales: " << data.X.size() << endl;
        //---------------------------------
        for(int j=0;j<MAX_ITER;j++){
            //Iteración j
            perceptron[i].testEveryX(data,i);
            perceptron[i].saveIfBestWV();
            if(perceptron[i].bestCotaFaults==0) break;
            perceptron[i].updateWV(data,i); //Falla aqui
            perceptron[i].resetFaults();
        }
        printf("Cota fallos: %f\n",perceptron[i].bestCotaFaults);
    }

    //Imprimir resultado
    printResult();

    saveResult();

    return 0;
}
