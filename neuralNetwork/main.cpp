#include <vector>
#include <initializer_list>
#include <stdint.h>
#include <cstdio>
#include <stdexcept>
#include <random>
#include <queue>
#include <iostream>
using namespace std;

using namespace std;
using VecDouble_t = vector<double>; //Neurona
/* W    [ w0(........) ]
        [ w1(........) ]
        [ ............ ]
        [ wn(........) ]*/
using MatDouble_t = vector<VecDouble_t>;    //Capa

double randDouble(double min, double max){
    static random_device dev;  //Coge numero de dispositivo hardware preparado para generacion de aleatorios
    static mt19937 rng(dev()); //Algoritmo pseudoaleatorio
    static uniform_real_distribution<double> dist(min,max);    //Distribucion lineal

    return dist(rng);
}

void fillVectorNoRandom(auto& vec, auto i, auto seed){
    size_t j=0;
    for (auto& v : vec){
        j++;
        v=(double)(seed*i*j)*0.1;
    }
}

void fillVectorRandom(auto& vec, double min, double max){
    for (auto& v : vec){
        v=randDouble(min,max);
    }
}

struct NeuralNetwork_t{
    explicit NeuralNetwork_t(initializer_list<uint16_t> const& layers,float learningR) {
        learningRate=learningR;
        //Al menos deben haber 2 capas
        if(layers.size()<2) throw out_of_range("Number of layers can not be less than 2");
        
        auto input_size = *layers.begin(); //Numero de entradas a la red

        for(auto it=layers.begin()+1; it!=layers.end() ; it++){
            //Capa it (it=puntero a un valor de layers)
            MatDouble_t matrix_w(*it);
            for(size_t i=0; i<*it ; i++){
                //Neurona i
                VecDouble_t neuron_w(input_size+1); //El 0 es el bias
                fillVectorRandom(neuron_w,-3.0,3.0);
                //fillVectorNoRandom(neuron_w,i+1,7);    //Solo para las pruebas

                matrix_w[i]=neuron_w;
            }
            m_layers.push_back(matrix_w);
            input_size=*it;
        }
    }

    VecDouble_t multiplyT(VecDouble_t const& input,MatDouble_t const& W) const{
        if(input.size()!=W[0].size()) throw length_error("Input and weight vector must have the same size.");

        VecDouble_t result(W.size(),0.0);
        for(size_t i=0; i<W.size() ; i++){
            //Neurona i de W
            for(size_t j=0 ; j<input.size() ; j++){
                //Peso j de la neurona i
                result[i]+=input[j]*W[i][j];
            }
        }

        return result;
    }

    constexpr auto sigmoid(auto x) const{
        if((1 + exp(-x))==0) throw out_of_range("Division entre 0 en funcion sigmoidea.");
        return 1 / (1 + exp(-x));
    }
    /*-------------------------------NUEVO--------------------------------*/
    constexpr auto sigmoidDeriv(auto x) const{  //Funciona
        return sigmoid(x)*(1-sigmoid(x));
    }

    constexpr auto signal(VecDouble_t const& x,auto layer,auto neuron) const{   //FUNCIONA
        double res=m_layers[layer][neuron][0];
        if(layer>0){
            for(size_t i=0;i<m_layers[layer-1].size();i++){
                res+=feedforwardinneuron(x,layer-1,i)*m_layers[layer][neuron][i+1];
            }
        }else{
            for(size_t i=0;i<x.size();i++){
                res+=x[i]*m_layers[layer][neuron][i+1];
            }
        }
        return res;
    }

    constexpr auto deltaOutputLayer(VecDouble_t const& x,VecDouble_t const y,auto layer,auto neuron) const{  //Funciona
        return errorDerivateFunction(feedforwardinneuron(x,layer,neuron),y[neuron],y.size())*sigmoidDeriv(signal(x,layer,neuron));
    }

    constexpr auto deltaHiddenLayers(VecDouble_t const& x,auto layer,auto neuron) const{    //Funciona
        //double m1=feedforwardinneuron(x,layer,neuron);
        double m1=signal(x,layer,neuron);   //No se si es esta o la linea comentada de arriba
        m1=sigmoidDeriv(m1);
        //if(sigmoidDeriv(signal(x,layer,neuron))!=(feedforwardinneuron(x,layer,neuron)*(1-feedforwardinneuron(x,layer,neuron)))) cout << "¡¡¡FALLA!!!" << endl;
        double m2=0;

        for(size_t i=0;i<m_layers[layer+1].size();i++){
            m2+=m_layers[layer+1][i][neuron+1]*deltaQueue.front()[i];
        }
        return m1*m2;
    }

    auto delta(VecDouble_t const& x,VecDouble_t const y,auto layer,auto neuron){    //Funciona
        //Posible optimizacion: Comprobar si ya tenemos el delta guardado en la cola antes de recalcular
        double delta;
        if(layer==m_layers.size()-1){
            delta=deltaOutputLayer(x,y,layer,neuron);
        }else{
            delta=deltaHiddenLayers(x,layer,neuron);
        }

        deltaQueue.back()[neuron]=delta;

        return delta;
    }

    //Cambiar para y multidimensional
    constexpr auto errorDerivateFunction(auto hx, auto y,auto size_y) const{    //Funciona
        return 2*(hx-y)/size_y;
    }

    //Cambiar para y multidimensional
    constexpr auto errorFunctionInNeuron(double hx, double y) const{
        return pow(hx-y,2);
    }

    auto errorFunction(const VecDouble_t hx,const VecDouble_t y) const{
        double error=0;
        for(size_t i=0;i<y.size();i++){
            error+=errorFunctionInNeuron(hx[i],y[i]);
        }
        error=error/y.size();
        return error;
    }

    //Cambiar para y multidimensional
    double errorFunctionVector(MatDouble_t const& X, MatDouble_t const& Y) const{
        double errorCont=0;

        for(size_t i=0; i<X.size() ; i++){
            errorCont+=errorFunction(feedforward(X[i]),Y[i]);
        }
        errorCont=errorCont/X.size();

        return errorCont;
    }

    //Derivada parcial para el peso m_layers[layer][neuron][beforeNeuron]
    double errorDerivateParcialFunction(VecDouble_t const& x,VecDouble_t const y,auto layer,auto neuron,auto beforeNeuron){    //Funciona
        double deltaV;
        if(beforeNeuron==0){
            return delta(x,y,layer,neuron);
        }else{
            beforeNeuron--;
            deltaV=delta(x,y,layer,neuron)*feedforwardinneuron(x,layer-1,beforeNeuron);
            return deltaV;
        }
    }

    //Devuelve vector de las derivadas parciales de la funcion de error de 1 neurona
    VecDouble_t errorDerivateParcialFunctions(VecDouble_t const& x,VecDouble_t const y,auto layer,auto neuron){    //Funciona
        VecDouble_t res(m_layers[layer][neuron].size());

        for(size_t i=0;i<res.size();i++){
            res[i]=errorDerivateParcialFunction(x,y,layer,neuron,i);
        }

        return res;
    }

    void multiplyIntVectors(auto n, auto &v) const{ //FUNCIONA
        for(size_t i=0;i<v.size();i++){
            v[i]=v[i]*n;
        }
    }

    void subVectors(auto &v1,auto &v2){ //FUNCIONA
        if(v1.size()!=v2.size()){
            throw length_error("Vectors must have the same size when sub.");
        }

        for(size_t i=0;i<v1.size();i++){
            v1[i]=v1[i]-v2[i];
        }
    }

    void multiplyIntMatrix(auto n, auto &v) const{   //Funciona
        for(size_t i=0;i<v.size();i++){
            for(size_t j=0;j<v[i].size();j++){
                for(size_t k=0;k<v[i][j].size();k++){
                    v[i][j][k]=v[i][j][k]*n;
                }
            }
        }
    }

    void subMatrix(auto &v1,auto &v2){   //Funciona
        if(v1.size()!=v2.size()){
            throw length_error("Vectors must have the same size when sub.");
        }

        for(size_t i=0;i<v1.size();i++){
            for(size_t j=0;j<v1[i].size();j++){
                for(size_t k=0;k<v1[i][j].size();k++){
                    v1[i][j][k]=v1[i][j][k]-v2[i][j][k];
                }
            }
        }
    }

    //Actualiza los pesos de una neurona
    VecDouble_t updateNeuron(VecDouble_t const& x,VecDouble_t const y,auto layer,auto neuron){    //Funciona
        VecDouble_t result = errorDerivateParcialFunctions(x,y,layer,neuron);

        return result;   //New

        /*multiplyIntVectors(learningRate,result);   //New

        subVectors(m_layers[layer][neuron],result);*/   //New
    }

    //Actualiza los pesos de una capa
    MatDouble_t updateLayer(VecDouble_t const& x,VecDouble_t const y,auto layer){
        MatDouble_t layerVector;   //New

        //Eliminamos el delta desfasado
        if(deltaQueue.size()>=2){
            deltaQueue.pop();
        }

        //Añadimos el delta de la capa actual
        VecDouble_t newDeltas(m_layers[layer].size());
        deltaQueue.push(newDeltas);

        for(size_t i=0;i<m_layers[layer].size();i++){
            //updateNeuron(x,y,layer,i);   //New
            layerVector.push_back(updateNeuron(x,y,layer,i));   //New
        }

        return layerVector;

    }

    //Actualiza los pesos de la red
    void updateWeights(VecDouble_t const& x,VecDouble_t const y){
        vector<MatDouble_t> layersVector(m_layers.size());   //New

        while(deltaQueue.size()>0) deltaQueue.pop();    //Reseteamos los deltas

        //De la ultima capa a la primera para backpropagation
        for(size_t i=m_layers.size()-1;(i+1)>0;i--){    //(i+1)>0 porque size_t no puede ser negativo // Es identico a i>=0
            //updateLayer(x,y,i);   //New
            layersVector[i]=updateLayer(x,y,i);   //New
        }

        multiplyIntMatrix(learningRate,layersVector);   //New
        subMatrix(m_layers,layersVector);   //New
    }

    VecDouble_t feedforwardinlayer(VecDouble_t const& x,auto layer) const{  //FUNCIONA
        //r1 = sigmoid(x*m_layers[0])
        //r2 = sigmoid(r1*m_layers[1])
        //...

        size_t i=0;
        VecDouble_t result(x);
        for (auto const& Wi : m_layers){
            //Capa Wi
            //Añadimos el x0 = 1
            result.resize(result.size()+1); //Aumentamos size
            copy(result.rbegin()+1, result.rend(), result.rbegin()); //Desplazamos los elementos 1 pos a la derecha
            result[0]=1.0;

            result = sigmoid(multiplyT(result,Wi));

            if(i==layer){
                return result;
            }
            i++;
        }

        return result;
    }

    double feedforwardinneuron(VecDouble_t const& x,auto layer,auto neuron) const{  //Funciona
        return feedforwardinlayer(x,layer)[neuron];
    }

    void train(MatDouble_t const& X,MatDouble_t const& Y,uint16_t epochs){
        if(X.size()!=Y.size()){
            throw length_error("Input and output vector must have the same size.");
        }

        double errorF;

        for(size_t i=0;i<epochs;i++){
            for(size_t j=0;j<X.size();j++){
                //cout << "Data " << j << endl;
                updateWeights(X[j],Y[j]);
            }
            errorF=errorFunctionVector(X,Y);
            cout << "Error cuadratico medio: " << errorF << endl;
            //¿Borrar?:
            if(errorF<0.1){
                cout << "Epocas necesarias: " << i << endl;
                break;   
            }
        }
    }
    /*--------------------------------------------------------------------*/

    VecDouble_t sigmoid (VecDouble_t const& vec) const{
        VecDouble_t result(vec.size(),0.0);

        for(size_t i=0; i<vec.size() ; i++){
            result[i]=sigmoid(vec[i]);
        }

        return result;
    }

    VecDouble_t feedforward(VecDouble_t const& x) const{
        //r1 = sigmoid(x*m_layers[0])
        //r2 = sigmoid(r1*m_layers[1])
        //...

        /*VecDouble_t result(x);
        for (auto const& Wi : m_layers){
            //Capa Wi
            //Añadimos el x0 = 1
            result.resize(result.size()+1); //Aumentamos size
            copy(result.rbegin()+1, result.rend(), result.rbegin()); //Desplazamos los elementos 1 pos a la derecha
            result[0]=1.0;

            result = sigmoid(multiplyT(result,Wi));
        }

        return result;*/
        return feedforwardinlayer(x,m_layers.size()-1);
        
    }


    MatDouble_t createCopyLayer(MatDouble_t const& layer){
        MatDouble_t matrixCopy;

        for(VecDouble_t v : layer){
            VecDouble_t newVec(v);
            matrixCopy.push_back(newVec);
        }
        return matrixCopy;
    }

    vector<MatDouble_t> getm_layers(){
        vector<MatDouble_t> res;
        for(MatDouble_t layer : m_layers){
            res.push_back(createCopyLayer(layer));
        }

        return res;
    }

private:
    vector<MatDouble_t> m_layers;
    queue<VecDouble_t> deltaQueue;
    float learningRate=0.2;
};

MatDouble_t X {
    {0.0, 0.0},
    {0.0, 1.0},
    {1.0, 0.0},
    {1.0, 1.0}
};

MatDouble_t Y {
    {0.0,0.0},
    {1.0,1.1},
    {1.0,1.1},
    {0.0,0.0}
};

double evaluateNet(NeuralNetwork_t& net,MatDouble_t const& X, VecDouble_t const& Y){
    int j=0;
    double fails=0;
    for (auto const& xi : X){
        //Probamos con la entrada xi (iterador j)
        auto res = net.feedforward(xi);

        fails += pow(Y[j]-res[0],2);
        
        j++;
    }

    return fails;
}

void randomTrain(initializer_list<uint16_t> layerStruct, NeuralNetwork_t& bestNet,MatDouble_t const& X, VecDouble_t const& Y,float learningR){
    double lessFails=pow(X.size(),2);
    double fails=0;


    for(size_t i=0;i<5000;i++){
        //Iteracion i
        NeuralNetwork_t net(layerStruct,learningR);
        
        fails=evaluateNet(net,X,Y);
        //Si es la mejor la guardamos
        if(fails<lessFails){
            lessFails=fails;
            bestNet=net;
        }

    }
}

void run(){
    initializer_list<uint16_t> layerStruct={2,3,3,2};
    float learningRate=0.15;
    NeuralNetwork_t net(layerStruct,learningRate);
    
    //randomTrain(layerStruct,net,X,Y);
    net.train(X,Y,10000);


    //Predecimos los valores
    auto res = net.feedforward({0.0 , 0.0});
    printf("0.0 xor 0.0 = %f -> %i\n",res[0],(int)(res[0]+0.5));
    res = net.feedforward({0.0 , 1.0});
    printf("0.0 xor 1.0 = %f -> %i\n",res[0],(int)(res[0]+0.5));
    res = net.feedforward({1.0 , 0.0});
    printf("1.0 xor 0.0 = %f -> %i\n",res[0],(int)(res[0]+0.5));
    res = net.feedforward({1.0 , 1.0});
    printf("1.0 xor 1.0 = %f -> %i\n",res[0],(int)(res[0]+0.5));
}

int main(){

    try{
        run();
    }catch(exception &e){
        printf("[EXCEPTION]: %s\n",e.what());
        return -1;
    }

    return 0;
}