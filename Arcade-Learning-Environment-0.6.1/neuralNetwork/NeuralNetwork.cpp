#ifndef NEURAL_NETWORK_CPP
#define NEURAL_NETWORK_CPP

#include "NeuralNetwork.h"


void multiplyIntVectors(auto n, auto &v){
    for(size_t i=0;i<v.size();i++){
        v[i]=v[i]*n;
    }
}

void subVectors(auto &v1,auto &v2){
    if(v1.size()!=v2.size()){
        throw length_error("Vectors must have the same size when sub.");
    }

    for(size_t i=0;i<v1.size();i++){
        v1[i]=v1[i]-v2[i];
    }
}

void multiplyIntMatrix(auto n, auto &v){
    for(size_t i=0;i<v.size();i++){
        for(size_t j=0;j<v[i].size();j++){
            for(size_t k=0;k<v[i][j].size();k++){
                v[i][j][k]=v[i][j][k]*n;
            }
        }
    }
}

void subMatrix(auto &v1,auto &v2){
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

NeuralNetwork_t::NeuralNetwork_t(){
    
}

NeuralNetwork_t::NeuralNetwork_t(initializer_list<uint16_t> const& layers,float learningR) {
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

    for(size_t i=0;i<m_layers.size();i++){
        functionsAct.push_back(ActF::SIGMOID);
    }
}

void NeuralNetwork_t::setActiveFunctions(initializer_list<ActF> v){
    size_t i=0;
    for(auto it=v.begin()+1; it!=v.end() ; it++){
        functionsAct[i]=*it;
        i++;
    }
}

//OPTIMIZACION: Devolver por referencia
VecDouble_t NeuralNetwork_t::multiplyT(VecDouble_t const& input,MatDouble_t const& W) const{
    if(input.size()!=W[0].size()){
        string msg="Input and weight vector must have the same size.\nInput size: " + to_string(input.size()) + "\nWeight size: " + to_string(W[0].size()) + "\n";
        throw length_error(msg);
    }

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

double NeuralNetwork_t::activeFunction(auto x, auto layer) const{
    switch(functionsAct[layer]){
        case ActF::SIGMOID: return sigmoid(x);
        case ActF::RELU: return relu(x);
        default: break;
    }

    return -1;
}

constexpr double NeuralNetwork_t::sigmoid(auto x) const{
    return 1 / (1 + exp(-x));
}
constexpr double NeuralNetwork_t::relu(double x) const{
    if(0>x) return 0;
    else return x;
}

double NeuralNetwork_t::activeFunctionDeriv(auto x, auto layer) const{
    switch(functionsAct[layer]){
        case ActF::SIGMOID: return sigmoidDeriv(x);
        case ActF::RELU: return reluDeriv(x);
        default: break;
    }

    return -1;
}

constexpr auto NeuralNetwork_t::sigmoidDeriv(auto x) const{
    return sigmoid(x)*(1-sigmoid(x));
}
constexpr auto NeuralNetwork_t::reluDeriv(auto x) const{
    if(x<0) return 0;
    else return 1;
}

//Meter en matriz al hacer forwardPropagation para evitar calculos innecesarios: OPTIMIZACION
/*constexpr auto NeuralNetwork_t::signal(VecDouble_t const& x,auto layer,auto neuron){
    double res=m_layers[layer][neuron][0];
    if(layer>0){
        for(size_t i=0;i<m_layers[layer-1].size();i++){
            res+=feedforwardMat[layer-1][i]*m_layers[layer][neuron][i+1];
        }
    }else{
        for(size_t i=0;i<x.size();i++){
            res+=x[i]*m_layers[layer][neuron][i+1];
        }
    }
    return res;
}*/

constexpr auto NeuralNetwork_t::deltaOutputLayer(VecDouble_t const& x,VecDouble_t const& y,auto layer,auto neuron){  //Funciona
    //Sesgado
    if(y[neuron]==1) return ((double)Yneg[neuron]/(Yneg[neuron]+Ypos[neuron]))*errorDerivateFunction(feedforwardMat[layer][neuron],y[neuron],y.size())*activeFunctionDeriv(signalMat[layer][neuron],layer);
    else return ((double)Ypos[neuron]/(Yneg[neuron]+Ypos[neuron]))*errorDerivateFunction(feedforwardMat[layer][neuron],y[neuron],y.size())*activeFunctionDeriv(signalMat[layer][neuron],layer);
    //Sin sesgar
    //return errorDerivateFunction(feedforwardMat[layer][neuron],y[neuron],y.size())*activeFunctionDeriv(signalMat[layer][neuron],layer);
}

constexpr auto NeuralNetwork_t::deltaHiddenLayers(VecDouble_t const& x,auto layer,auto neuron){    //Funciona
    double m1=signalMat[layer][neuron];   //No se si es esta o la linea comentada de arriba
    m1=activeFunctionDeriv(m1,layer);
    double m2=0;

    for(size_t i=0;i<m_layers[layer+1].size();i++){
        m2+=m_layers[layer+1][i][neuron+1]*deltaQueue.front()[i];
    }
    return m1*m2;
}

auto NeuralNetwork_t::delta(VecDouble_t const& x,VecDouble_t const& y,auto layer,auto neuron){    //Funciona
    double delta;
    if(layer==m_layers.size()-1){
        delta=deltaOutputLayer(x,y,layer,neuron);
    }else{
        delta=deltaHiddenLayers(x,layer,neuron);
    }

    deltaQueue.back()[neuron]=delta;

    return delta;
}

constexpr auto NeuralNetwork_t::errorDerivateFunction(auto hx, auto y,auto size_y) const{    //Funciona
    //(m-n)/m
    return 2*(hx-y)/size_y;
}

constexpr auto NeuralNetwork_t::errorFunctionInNeuron(double hx, double y) const{
    //(m-n)/m
    return pow(hx-y,2);
}

auto NeuralNetwork_t::errorFunction(const VecDouble_t hx,const VecDouble_t y) const{
    double error=0;
    for(size_t i=0;i<y.size();i++){
        //Sesgado
        if(y[i]==1) error+=((double)Yneg[i]/(Yneg[i]+Ypos[i]))*errorFunctionInNeuron(hx[i],y[i]);
        else error+=((double)Ypos[i]/(Yneg[i]+Ypos[i]))*errorFunctionInNeuron(hx[i],y[i]);
        //Sin sesgar
        //error+=errorFunctionInNeuron(hx[i],y[i]);
    }
    error=error/y.size();
    return error;
}

double NeuralNetwork_t::errorFunctionVector(MatDouble_t const& X, MatDouble_t const& Y){
    double errorCont=0;

    for(size_t i=0; i<X.size() ; i++){
        errorCont+=errorFunction(feedforward(X[i]),Y[i]);
    }
    errorCont=errorCont/X.size();

    return errorCont;
}

//Derivada parcial para el peso m_layers[layer][neuron][beforeNeuron]
double NeuralNetwork_t::errorDerivateParcialFunction(VecDouble_t const& x,VecDouble_t const& y,auto layer,auto neuron,auto beforeNeuron){    //Funciona
    double deltaV;
    if(beforeNeuron==0){
        return delta(x,y,layer,neuron);
    }else{
        beforeNeuron--;
        if(layer>0){
            deltaV=delta(x,y,layer,neuron)*feedforwardMat[layer-1][beforeNeuron];
        }else{
            deltaV=delta(x,y,layer,neuron)*x[beforeNeuron];
        }
        return deltaV;
    }
}

//Devuelve vector de las derivadas parciales de la funcion de error de 1 neurona
//Devolver por referencia: OPTIMIZACION
VecDouble_t NeuralNetwork_t::errorDerivateParcialFunctions(VecDouble_t const& x,VecDouble_t const& y,auto layer,auto neuron){    //Funciona
    VecDouble_t res(m_layers[layer][neuron].size());

    for(size_t i=0;i<res.size();i++){
        res[i]=errorDerivateParcialFunction(x,y,layer,neuron,i);
    }

    return res;
}

//Actualiza los pesos de una capa
//Devolver por referencia: OPTIMIZACION
MatDouble_t NeuralNetwork_t::updateLayer(VecDouble_t const& x,VecDouble_t const& y,auto layer){
    MatDouble_t layerVector;

    //Eliminamos el delta desfasado
    if(deltaQueue.size()>=2){
        deltaQueue.pop();
    }

    //Añadimos el delta de la capa actual
    VecDouble_t newDeltas(m_layers[layer].size());
    deltaQueue.push(newDeltas);

    for(size_t i=0;i<m_layers[layer].size();i++){
        layerVector.push_back(errorDerivateParcialFunctions(x,y,layer,i));
    }

    return layerVector;

}

//Actualiza los pesos de la red
void NeuralNetwork_t::updateWeights(VecDouble_t const& x,VecDouble_t const& y){
    vector<MatDouble_t> layersVector(m_layers.size());

    calculateFeedForwardMat(x);

    while(deltaQueue.size()>0) deltaQueue.pop();    //Reseteamos los deltas

    //De la ultima capa a la primera para backpropagation
    for(size_t i=m_layers.size()-1;(i+1)>0;i--){    //(i+1)>0 porque size_t no puede ser negativo // Es identico a i>=0
        layersVector[i]=updateLayer(x,y,i);
    }

    multiplyIntMatrix(learningRate,layersVector);
    subMatrix(m_layers,layersVector);
}

void NeuralNetwork_t::calculateFeedForwardMat(VecDouble_t const& x){
    feedforwardMat.resize(0);
    signalMat.resize(0);
    for(size_t i=0; i< m_layers.size();i++){
        VecDouble_t v(m_layers[i].size());
        for(size_t j=0; j<m_layers[i].size();j++){
            v[j]=999999;
        }
        feedforwardMat.push_back(v);
        signalMat.push_back(v);
    }

    size_t i=0;
    VecDouble_t result(x);
    for (auto const& Wi : m_layers){
        //Capa Wi
        
        
        //Añadimos el x0 = 1
        result.resize(result.size()+1); //Aumentamos size
        copy(result.rbegin()+1, result.rend(), result.rbegin()); //Desplazamos los elementos 1 pos a la derecha
        result[0]=1.0;
        

        signalMat[i]=multiplyT(result,Wi);
        result = activeFunction(signalMat[i],i);
        feedforwardMat[i]=result;
        
        i++;
    }
}

//Devolver por referencia: OPTIMIZACION
VecDouble_t NeuralNetwork_t::feedforwardinlayer(VecDouble_t const& x,auto layer){
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

        result = activeFunction(multiplyT(result,Wi),i);
        

        if(i==layer){
            return result;
        }
        i++;
    }
    return result;
    
}

double NeuralNetwork_t::feedforwardinneuron(VecDouble_t const& x,auto layer,auto neuron){  //Funciona
    return feedforwardinlayer(x,layer)[neuron];
}

void NeuralNetwork_t::train(MatDouble_t const& X,MatDouble_t const& Y,MatDouble_t const& Xval,MatDouble_t const& Yval,uint16_t epochs){

    if(X.size()!=Y.size()){
        throw length_error("Input and output vector must have the same size.");
    }
    //Sin comprobar---------------------------
    for(size_t i=0;i<Y[0].size();i++){
        Yneg.push_back(0);
        Ypos.push_back(0);
    }
    for(size_t i=0;i<Y.size();i++){
        
        for(size_t j=0;j<Y[i].size();j++){
            if(Y[i][j]==1) Ypos[j]++;
            else Yneg[j]++;
        }
    }
    //-----------------------------------------
    double errorF;

    for(size_t i=0;i<epochs;i++){
        for(size_t j=0;j<X.size();j++){
            updateWeights(X[j],Y[j]);
        }
        cout << "Epoca " << i << endl;
        errorF=errorFunctionVector(X,Y);
        cout << "Error cuadratico medio: " << errorF << endl;
        errorF=errorFunctionVector(Xval,Yval);
        cout << "Error cuadratico medio validation: " << errorF << endl;
        //¿Borrar?:
        /*if(errorF<0.1){
            cout << "Epocas necesarias: " << i << endl;
            break;   
        }*/
    }
    errorF=errorFunctionVector(Xval,Yval);
    cout << "Error cuadratico medio: " << errorF << endl;
}

//Devolver por referencia: OPTIMIZACION
VecDouble_t NeuralNetwork_t::activeFunction (VecDouble_t const& vec, auto layer){
    switch(functionsAct[layer]){
        case ActF::SIGMOID: return sigmoid(vec);
        case ActF::RELU: return relu(vec);
        default: break;
    }

    VecDouble_t v;
    return v;
}

//Devolver por referencia: OPTIMIZACION
VecDouble_t NeuralNetwork_t::relu (VecDouble_t const& vec) const{
    VecDouble_t result(vec.size(),0.0);

    for(size_t i=0; i<vec.size() ; i++){
        result[i]=relu(vec[i]);
    }

    return result;
}

//Devolver por referencia: OPTIMIZACION
VecDouble_t NeuralNetwork_t::sigmoid (VecDouble_t const& vec) const{
    VecDouble_t result(vec.size(),0.0);

    for(size_t i=0; i<vec.size() ; i++){
        result[i]=sigmoid(vec[i]);
    }

    return result;
}

//Devolver por referencia: OPTIMIZACION
VecDouble_t NeuralNetwork_t::feedforward(VecDouble_t const& x){
    //r1 = sigmoid(x*m_layers[0])
    //r2 = sigmoid(r1*m_layers[1])
    //...

    return feedforwardinlayer(x,m_layers.size()-1);
    
}


MatDouble_t NeuralNetwork_t::createCopyLayer(MatDouble_t const& layer){
    MatDouble_t matrixCopy;

    for(VecDouble_t v : layer){
        VecDouble_t newVec(v);
        matrixCopy.push_back(newVec);
    }
    return matrixCopy;
}

vector<MatDouble_t> NeuralNetwork_t::getm_layers(){
    vector<MatDouble_t> res;
    for(MatDouble_t layer : m_layers){
        res.push_back(createCopyLayer(layer));
    }

    return res;
}

double NeuralNetwork_t::evaluateNet(MatDouble_t const& X, VecDouble_t const& Y){
    int j=0;
    double fails=0;
    for (auto const& xi : X){
        //Probamos con la entrada xi (iterador j)
        auto res = feedforward(xi);

        fails += pow(Y[j]-res[0],2);
        
        j++;
    }

    return fails;
}

void NeuralNetwork_t::save(const std::string s) const{

    std::ofstream ficheroSalida;
    ficheroSalida.open (s);
    ficheroSalida << "[";
    for(size_t i=0;i<m_layers.size();i++){
        ficheroSalida << "[";
        for(size_t j=0;j<m_layers[i].size();j++){
            ficheroSalida << "[";
            for(size_t k=0;k<m_layers[i][j].size();k++){
                ficheroSalida << m_layers[i][j][k];

                if(m_layers[i][j].size()-1>k)
                    ficheroSalida <<",";
            }
            ficheroSalida << "]";
        }
        ficheroSalida << "]";
    }
    ficheroSalida << "]";
    
    ficheroSalida.close();

}

void NeuralNetwork_t::load(const std::string fichero){
    m_layers.resize(0);
    MatDouble_t y;
    VecDouble_t z;
    string s;
    s="";

    ifstream ficheroEntrada;
    char letra;

    ficheroEntrada.open (fichero);
    ficheroEntrada >> letra;
    while (! ficheroEntrada.eof() ) {
        ficheroEntrada >> letra;
        while(letra!=']'){
            while(letra!=']'){
                ficheroEntrada >> letra;
                if(letra=='['){
                    ficheroEntrada >> letra;
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
            ficheroEntrada >> letra;
        }
        if(y.size()>0){
            m_layers.push_back(y);
            y.clear();
        }
    }
    ficheroEntrada.close();
    //escribir(x,"leida");
    functionsAct.resize(0);
    for(size_t i=0;i<m_layers.size();i++){
        functionsAct.push_back(ActF::SIGMOID);
    }
}

#endif