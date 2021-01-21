/*
 int sum = 0;       //CANTIDAD DE NÚMEROS PARES QUE HAY EN UNA MATRIZ
    
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            sum += !(matrix[i][j] & 1);
        }
    }    

    void ejemploAtoi(){
        char *cad = "123455";
        int num   = atoi(cad);
    }
*/

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

typedef int** Matrix;

int convertCharToNumber(char);
Matrix allocateMatrix(int);
bool isNumber(char);
void printMatrix(Matrix, int);

int main() {
    FILE *f = NULL;
    Matrix matrix = NULL;
    int tam = 0, i = 0, j = 0;
    int numero;
    char fileName[] = "prueba.txt";

    f = fopen(fileName, "r");
    if(f == NULL){
        printf("Error al abrir el archivo\n");
    }
    else{
        tam = convertCharToNumber(getc(f));

        // Allocate memory for matrix
        matrix = allocateMatrix(tam);
        
        // Read matrix from file
        while(!feof(f)){
            numero = getc(f);
            if(isNumber(numero)) {
                matrix[i][j] = convertCharToNumber(numero);
                j++;
            }
            if (j == tam) {
                i++; 
                j = 0;
            }
        }

        printMatrix(matrix, tam);
    }

    return 0;
}

int convertCharToNumber(char num) {
    return num - '0';
}

bool isNumber(char numero) {
    return numero >= '0' && numero <= '9';
}

void printMatrix(Matrix m, int size) { 
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++)
            printf("%d ", m[i][j]);
        printf("\n");
    }
}

Matrix allocateMatrix(int size) {
    Matrix matrix = (int **)malloc(sizeof(int*) * size);

    for (int i = 0; i < size; i++)
        matrix[i] = (int *)malloc(sizeof(int) * size);  
    
    return matrix;
}