#include <Utils/Balance.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <tuple>
#include <cstring>
#include <string>
#include <fstream>
#include <iostream>

constexpr uint8_t KNumInputs = 6;
using Vec_d = std::vector<double>;
using Vec_st = std::vector<std::size_t>;
using Mat_d = std::vector<Vec_d>;
using Vec_s = std::vector<std::string>;

//cuenta las lineas de un fichero
int contador(std::string const& nameFile){

	int i=0;
	std::string line;
	std::ifstream fichero(nameFile);
	while (std::getline(fichero, line)) {
		i++;
	}

	fichero.close();

	return i;
}

//lee datos de las partidas
std::tuple<Mat_d, Mat_d> leerDatos(std::string const& nameFile,Vec_st &numCasos){

	int count = contador(nameFile);
    Mat_d X(count);
	Mat_d y(count);
	std::ifstream fichero(nameFile);
    std::string line;
    int numLine=0;
    while (std::getline(fichero, line)) {
		X[numLine].push_back(1);
        int init=0;
		int end=0;
		int count=0;
		while( end = line.find(";", init), end >= 0 )
		{
			double n = atof(line.substr(init, end - init).c_str());
			if(count>=59){
				if(n==0){
					y[numLine].push_back(-1);
				}else{
					y[numLine].push_back(n);
					numCasos[count-59]+=1;
				}
			}else{
            	X[numLine].push_back(n);
			}
			init = end + 1;
			count++;
		}
		double n = atof(line.substr(init).c_str());
		if(n==0){
			y[numLine].push_back(-1);
		}else{
			y[numLine].push_back(n);
			numCasos[4]+=1;
		}

		numLine++;
    }

	fichero.close();

	return {X,y};
}

//funcion hipotesis
double h(Vec_d const& xi,Vec_d const& w){
	assert(xi.size() == w.size());
	double result {0.0};
	
	for(std::size_t i=0;i<xi.size(); i++){
		result+=xi[i]*w[i];
	}
	return (result >0 ) ? 1.0 : -1.0;
}

//calculo error
Vec_st calculateError(Mat_d const& X, Vec_d const& y, Vec_d const& w){
	assert(y.size() == X.size());
	
	Vec_st errors;
	errors.reserve(X.size() / 4); 
	
	for(std::size_t i=0;i<X.size(); i++){
		auto& xi = X[i];
		auto& yi = y[i];
		
		if (yi != h(xi, w) )
			errors.push_back(i);
	}
	
	return errors;
}

//ajustar w
void adjust_w(Vec_d& w, Vec_d const& xi, double yi) {
	assert(xi.size() == w.size());
	
	for(std::size_t i=0;i<xi.size(); i++){
		w[i] += xi[i] * yi;
	}
}

//entrenamiento
std::tuple<Vec_d, uint64_t, uint64_t> train(Mat_d const& X, Vec_d const& y, uint64_t const& maxiter){
	assert(y.size() == X.size());
	assert(X.size() >= 1);
	
	Vec_d w(X[0].size() );
	Vec_d best_w {w};
	uint64_t best_w_errors {X.size()+1};
	uint64_t i {0};
	do{
		//calcular errores y actualizar b_w
		auto errors = calculateError(X,y,w);
		if(errors.size()<best_w_errors){
			best_w=w;
			best_w_errors=errors.size();
		}
		//salis si 0 errores
		if(best_w_errors==0) break;
		//actualizar w con un error aleatorio
		std::size_t rndidx = errors [ std:: rand() % errors.size() ];
		adjust_w(w, X[rndidx], y[rndidx]);
		
	}while(++i < maxiter);

	return {best_w,best_w_errors,i};
}

void printVector(Vec_d const& v){
	std::printf("( ");
	for(auto const&vi: v){
		std::printf(" %f ", vi);
	}
	std::puts(")");
}

//muestra la matriz por pantalla
void printMatrix(Mat_d const& v){
	std::printf("{ ");
	for(auto const&vi: v){
		printVector(vi);
	}
	std::puts("}");
}

//Devuelve la columna de una mtriz
Vec_d colMatrix(Mat_d const& m,int const& col){
	Vec_d v;
	for(size_t i=0;i<m.size();i++){
		v.push_back(m[i][col]);
	}

	return v;
}

//Escribe en un fichero una matriz
void write(std::string const& nameFile, Mat_d const& w){

	std::ofstream outfile;

    outfile.open(nameFile, std::ios_base::app);

    outfile << "{";
    for(size_t i=0;i<w.size();i++){
        outfile << "{";
        for(size_t  j=0;j<w[i].size();j++){
            outfile << w[i][j];
			if(w[i].size()-1>j)
            	outfile  << ',';
        }
        outfile << "}";

        if(w.size()-1>i)
            outfile  << ',';
    }
    outfile << "}\n";
}

int main (int argc, char**argv){


	Vec_st numCasos={0,0,0,0,0};
	auto [X,Y]=leerDatos(argv[1],numCasos);
	for(size_t i=0;i<numCasos.size();i++){
		std::cout << numCasos[i]<< std::endl;
	}
	//std::cout << "X: "<<X.size() << std::endl;

	//generamos datos de la clase minoritaria
	Balance b(numCasos);
	auto [X1,Y1]=b.generate(X,Y);
	std::cout <<"X1: "<< X1.size() << std::endl;
	auto [X2,Y2]=b.generate(X1,Y1);
	std::cout <<"X2: "<< X2.size() << std::endl;
	auto [X3,Y3]=b.generate(X2,Y2);
	std::cout <<"X3: "<< X3.size() << std::endl;
	
	//entrenamos las 5 salidas
    auto [w,be,it]=train(X3,colMatrix(Y3,0),100);
	auto [w1,be1,it1]=train(X3,colMatrix(Y3,1),100);
	auto [w2,be2,it2]=train(X3,colMatrix(Y3,2),100);
	auto [w3,be3,it3]=train(X3,colMatrix(Y3,3),100);
	auto [w4,be4,it4]=train(X3,colMatrix(Y3,4),100);
	Mat_d m;
	m.push_back(w);
	m.push_back(w1);
	m.push_back(w2);
	m.push_back(w3);
	m.push_back(w4);
	printMatrix(m);

	//escribimos en el fichero los vectores de pesos
	write("IA.txt",m);
	
	return 0;
}