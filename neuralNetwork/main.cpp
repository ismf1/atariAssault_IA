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

void fillVectorRandom(auto& vec, double min, double max){
    for (auto& v : vec){
        v=randDouble(min,max);
    }
}

struct NeuralNetwork_t{
    explicit NeuralNetwork_t(initializer_list<uint16_t> const& layers) {
        //Al menos deben haber 2 capas
        if(layers.size()<2) throw out_of_range("Number of layers can not be less than 2");
        
        auto input_size = *layers.begin(); //Numero de entradas a la red

        for(auto it=layers.begin()+1; it!=layers.end() ; it++){
            //Capa it (it=puntero a un valor de layers)
            MatDouble_t matrix_w(*it);
            for(size_t i=0; i<*it ; i++){
                //Neurona i
                VecDouble_t neuron_w(input_size+1); //El 0 es el bias
                fillVectorRandom(neuron_w,-10.0,10.0);

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
        return 1 / (1 + exp(-x));
    }
    /*-------------------------------NUEVO--------------------------------*/
    constexpr auto sigmoidDeriv(auto x) const{  //Creo que funciona
        return sigmoid(x)*(1-x);
    }

    constexpr auto signal(VecDouble_t const& x,auto layer,auto neuron) const{   //FUNCIONA
        double res=m_layers[layer][neuron][0];
        for(size_t i=0;i<m_layers[layer-1].size();i++){
            res+=feedforwardinneuron(x,layer-1,i)*m_layers[layer][neuron][i+1];
        }
        return res;
    }

    constexpr auto deltaOutputLayer(VecDouble_t const& x,double const y,auto layer,auto neuron) const{  //Creo que funciona
        return errorDerivateFunction(feedforwardinneuron(x,layer,neuron),y)*sigmoidDeriv(signal(x,layer,neuron));
    }

    constexpr auto deltaHiddenLayers(VecDouble_t const& x,auto layer,auto neuron) const{    //Creo que funciona
        double m1=sigmoidDeriv(feedforwardinneuron(x,layer,neuron));
        double m2=0;

        for(size_t i=0;i<m_layers[layer+1].size();i++){
            m2+=m_layers[layer+1][i][neuron+1]*deltaQueue.front()[i];
        }
        return m1*m2;
    }

    auto delta(VecDouble_t const& x,double const y,auto layer,auto neuron){    //Creo que funciona
        double delta;
        if(layer==m_layers.size()-1){
            delta=deltaOutputLayer(x,y,layer,neuron);
        }else{
            delta=deltaHiddenLayers(x,layer,neuron);
        }

        deltaQueue.back()[neuron]=delta;

        return delta;
    }

    constexpr auto errorDerivateFunction(auto hx, auto y) const{    //Creo que funciona
        return 2*(hx-y);
    }

    //Esto solo vale si la dimension de Y es 1
    constexpr auto errorFunction(auto hx, auto y) const{
        return pow(hx-y,2);
    }

    /*constexpr auto errorFunctionVector(MatDouble_t const& X, VecDouble_t const& Y) const{
        uint16_t errorCont=0;

        //FALTA

        return -1;
    }*/

    //Derivada parcial para el peso m_layers[layer][neuron][beforeNeuron]
    double errorDerivateParcialFunction(VecDouble_t const& x,double const y,auto layer,auto neuron,auto beforeNeuron){    //Creo que funciona
        return delta(x,y,layer,neuron)*feedforwardinneuron(x,layer-1,beforeNeuron);
    }

    //Devuelve vector de las derivadas parciales de la funcion de error de 1 neurona
    VecDouble_t errorDerivateParcialFunctions(VecDouble_t const& x,double const y,auto layer,auto neuron){    //Creo que funciona
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

    void multiplyIntMatrix(auto n, auto &v) const{   //New
        for(size_t i=0;i<v.size();i++){
            for(size_t j=0;j<v[i].size();j++){
                for(size_t k=0;k<v[i][j].size();k++){
                    v[i][j][k]=v[i][j][k]*n;
                }
            }
        }
    }

    void subMatrix(auto &v1,auto &v2){   //New
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
    VecDouble_t updateNeuron(VecDouble_t const& x,double const y,auto layer,auto neuron){    //PROBLEMA: Primero actualizamos y luego usamos el valor de delta anterior
        VecDouble_t result = errorDerivateParcialFunctions(x,y,layer,neuron);

        return result;   //New

        /*multiplyIntVectors(learningRate,result);   //New

        subVectors(m_layers[layer][neuron],result);*/   //New
    }

    //Actualiza los pesos de una capa
    MatDouble_t updateLayer(VecDouble_t const& x,double const y,auto layer){
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
    void updateWeights(VecDouble_t const& x,double const y){
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

    double feedforwardinneuron(VecDouble_t const& x,auto layer,auto neuron) const{
        return feedforwardinlayer(x,layer)[neuron];
    }

    void train(MatDouble_t const& X,VecDouble_t const& Y,uint16_t epochs){
        if(X.size()!=Y.size()){
            throw length_error("Input and output vector must have the same size.");
        }

        for(size_t i=0;i<epochs;i++){
            for(size_t j=0;j<X.size();j++){
                //cout << "Data " << j << endl;
                updateWeights(X[j],Y[j]);
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
    float learningRate=0.1;
};

MatDouble_t X {
    {0.0, 0.0},
    {0.0, 1.0},
    {1.0, 0.0},
    {1.0, 1.0}
};

VecDouble_t Y {
    0.0,
    1.0,
    1.0,
    0.0
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

void randomTrain(initializer_list<uint16_t> layerStruct, NeuralNetwork_t& bestNet,MatDouble_t const& X, VecDouble_t const& Y){
    double lessFails=pow(X.size(),2);
    double fails=0;


    for(size_t i=0;i<5000;i++){
        //Iteracion i
        NeuralNetwork_t net(layerStruct);
        
        fails=evaluateNet(net,X,Y);
        //Si es la mejor la guardamos
        if(fails<lessFails){
            lessFails=fails;
            bestNet=net;
        }

    }
}

void run(){
    initializer_list<uint16_t> layerStruct={2,3,1};
    NeuralNetwork_t net(layerStruct);
    
    //randomTrain(layerStruct,net,X,Y);
    net.train(X,Y,200);

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