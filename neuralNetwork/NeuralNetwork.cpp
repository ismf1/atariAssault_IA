#include "NeuralNetwork.h"

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

NeuralNetwork_t::NeuralNetwork_t(initializer_list<uint16_t> const& layers,float learningR) {
    for(auto it=layers.begin()+1; it!=layers.end() ; it++){
        functionsAct.push_back(ActF::SIGMOID);
    }
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
    
    if(m_layers.size()!=functionsAct.size()) throw length_error("En el constructor: La lista de funciones de activacion debe ser del tamaño del numero de capas-1");
}

void NeuralNetwork_t::setActiveFunctions(initializer_list<ActF> v){
    if (v.size()!=functionsAct.size()) throw length_error("La lista de funciones de activacion debe ser del tamaño del numero de capas-1");
    size_t i=0;
    for(auto it=v.begin(); it!=v.end() ; it++){
        functionsAct[i]=*it;
        i++;
    }
}

VecDouble_t NeuralNetwork_t::multiplyT(VecDouble_t const& input,MatDouble_t const& W) const{
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

double NeuralNetwork_t::activeFunction(auto x, auto layer) const{
    switch(functionsAct[layer]){
        case ActF::SIGMOID: return sigmoid(x);
        case ActF::RELU: return relu(x);
        default: break;
    }
}

constexpr double NeuralNetwork_t::sigmoid(auto x) const{
    return 1 / (1 + exp(-x));
}
constexpr double NeuralNetwork_t::relu(double x) const{
    if(0>x) return 0;
    else return x;
}
/*-------------------------------NUEVO--------------------------------*/
double NeuralNetwork_t::activeFunctionDeriv(auto x, auto layer) const{
    switch(functionsAct[layer]){
        case ActF::SIGMOID: return sigmoidDeriv(x);
        case ActF::RELU: return reluDeriv(x);
        default: break;
    }
    return -1;
}

constexpr auto NeuralNetwork_t::sigmoidDeriv(auto x) const{  //Funciona
    return sigmoid(x)*(1-sigmoid(x));
}
constexpr auto NeuralNetwork_t::reluDeriv(auto x) const{
    if(x<0) return 0;
    else return 1;
}

constexpr auto NeuralNetwork_t::signal(VecDouble_t const& x,auto layer,auto neuron){   //FUNCIONA
    double res=m_layers[layer][neuron][0];
    if(layer>0){
        for(size_t i=0;i<m_layers[layer-1].size();i++){
            //res+=feedforwardinneuron(x,layer-1,i)*m_layers[layer][neuron][i+1];
            res+=feedforwardMat[layer-1][i]*m_layers[layer][neuron][i+1];
        }
    }else{
        for(size_t i=0;i<x.size();i++){
            res+=x[i]*m_layers[layer][neuron][i+1];
        }
    }
    return res;
}

constexpr auto NeuralNetwork_t::deltaOutputLayer(VecDouble_t const& x,VecDouble_t const y,auto layer,auto neuron){  //Funciona
    //return errorDerivateFunction(feedforwardinneuron(x,layer,neuron),y[neuron],y.size())*sigmoidDeriv(signal(x,layer,neuron));
    return errorDerivateFunction(feedforwardMat[layer][neuron],y[neuron],y.size())*activeFunctionDeriv(signal(x,layer,neuron),layer);
}

constexpr auto NeuralNetwork_t::deltaHiddenLayers(VecDouble_t const& x,auto layer,auto neuron){    //Funciona
    //double m1=feedforwardinneuron(x,layer,neuron);
    double m1=signal(x,layer,neuron);   //No se si es esta o la linea comentada de arriba
    m1=activeFunctionDeriv(m1,layer);
    //if(sigmoidDeriv(signal(x,layer,neuron))!=(feedforwardinneuron(x,layer,neuron)*(1-feedforwardinneuron(x,layer,neuron)))) cout << "¡¡¡FALLA!!!" << endl;
    double m2=0;

    for(size_t i=0;i<m_layers[layer+1].size();i++){
        m2+=m_layers[layer+1][i][neuron+1]*deltaQueue.front()[i];
    }
    return m1*m2;
}

auto NeuralNetwork_t::delta(VecDouble_t const& x,VecDouble_t const y,auto layer,auto neuron){    //Funciona
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
constexpr auto NeuralNetwork_t::errorDerivateFunction(auto hx, auto y,auto size_y) const{    //Funciona
    return 2*(hx-y)/size_y;
}

//Cambiar para y multidimensional
constexpr auto NeuralNetwork_t::errorFunctionInNeuron(double hx, double y) const{
    return pow(hx-y,2);
}

auto NeuralNetwork_t::errorFunction(const VecDouble_t hx,const VecDouble_t y) const{
    double error=0;
    for(size_t i=0;i<y.size();i++){
        error+=errorFunctionInNeuron(hx[i],y[i]);
    }
    error=error/y.size();
    return error;
}

//Cambiar para y multidimensional
double NeuralNetwork_t::errorFunctionVector(MatDouble_t const& X, MatDouble_t const& Y){
    double errorCont=0;

    for(size_t i=0; i<X.size() ; i++){
        errorCont+=errorFunction(feedforward(X[i]),Y[i]);
    }
    errorCont=errorCont/X.size();

    return errorCont;
}

//Derivada parcial para el peso m_layers[layer][neuron][beforeNeuron]
double NeuralNetwork_t::errorDerivateParcialFunction(VecDouble_t const& x,VecDouble_t const y,auto layer,auto neuron,auto beforeNeuron){    //Funciona
    double deltaV;
    if(beforeNeuron==0){
        return delta(x,y,layer,neuron);
    }else{
        beforeNeuron--;
        //deltaV=delta(x,y,layer,neuron)*feedforwardinneuron(x,layer-1,beforeNeuron);
        if(layer>0){
            deltaV=delta(x,y,layer,neuron)*feedforwardMat[layer-1][beforeNeuron];
        }else{
            deltaV=delta(x,y,layer,neuron)*x[beforeNeuron];
        }
        return deltaV;
    }
}

//Devuelve vector de las derivadas parciales de la funcion de error de 1 neurona
VecDouble_t NeuralNetwork_t::errorDerivateParcialFunctions(VecDouble_t const& x,VecDouble_t const y,auto layer,auto neuron){    //Funciona
    VecDouble_t res(m_layers[layer][neuron].size());

    for(size_t i=0;i<res.size();i++){
        res[i]=errorDerivateParcialFunction(x,y,layer,neuron,i);
    }

    return res;
}


//Actualiza los pesos de una capa
MatDouble_t NeuralNetwork_t::updateLayer(VecDouble_t const& x,VecDouble_t const y,auto layer){
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
        layerVector.push_back(errorDerivateParcialFunctions(x,y,layer,i));   //New
    }

    return layerVector;

}

//Actualiza los pesos de la red
void NeuralNetwork_t::updateWeights(VecDouble_t const& x,VecDouble_t const y){
    vector<MatDouble_t> layersVector(m_layers.size());   //New

    calculateFeedForwardMat(x);

    while(deltaQueue.size()>0) deltaQueue.pop();    //Reseteamos los deltas

    //De la ultima capa a la primera para backpropagation
    for(size_t i=m_layers.size()-1;(i+1)>0;i--){    //(i+1)>0 porque size_t no puede ser negativo // Es identico a i>=0
        //updateLayer(x,y,i);   //New
        layersVector[i]=updateLayer(x,y,i);   //New
    }

    multiplyIntMatrix(learningRate,layersVector);   //New
    subMatrix(m_layers,layersVector);   //New
}

void NeuralNetwork_t::calculateFeedForwardMat(VecDouble_t const& x){
    feedforwardMat.resize(0);
    for(size_t i=0; i< m_layers.size();i++){
        VecDouble_t v(m_layers[i].size());
        for(size_t j=0; j<m_layers[i].size();j++){
            v[j]=999999;
        }
        feedforwardMat.push_back(v);
    }

    size_t i=0;
    VecDouble_t result(x);
    for (auto const& Wi : m_layers){
        //Capa Wi
        
        
        //Añadimos el x0 = 1
        result.resize(result.size()+1); //Aumentamos size
        copy(result.rbegin()+1, result.rend(), result.rbegin()); //Desplazamos los elementos 1 pos a la derecha
        result[0]=1.0;

        result = activeFunction(multiplyT(result,Wi),i);
        feedforwardMat[i]=result;
        /*VecDouble_t r(result);
        feedforwardMat[i] = r;*/
        //feedforwardMat[i] = result;
        
        i++;
    }
}

VecDouble_t NeuralNetwork_t::feedforwardinlayer(VecDouble_t const& x,auto layer){  //FUNCIONA
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

void NeuralNetwork_t::train(MatDouble_t const& X,MatDouble_t const& Y,uint16_t epochs){
    if(X.size()!=Y.size()){
        throw length_error("Input and output vector must have the same size.");
    }

    double errorF;

    for(size_t i=0;i<epochs;i++){
        for(size_t j=0;j<X.size();j++){
            //cout << "Data " << j << endl;
            updateWeights(X[j],Y[j]);
        }
        cout << "Epoca " << i << endl;
        errorF=errorFunctionVector(X,Y);
        cout << "Error cuadratico medio: " << errorF << endl;
        //¿Borrar?:
        /*if(errorF<0.1){
            cout << "Epocas necesarias: " << i << endl;
            break;   
        }*/
    }
    errorF=errorFunctionVector(X,Y);
    cout << "Error cuadratico medio: " << errorF << endl;
}
/*--------------------------------------------------------------------*/
VecDouble_t NeuralNetwork_t::activeFunction (VecDouble_t const& vec, auto layer){
    switch(functionsAct[layer]){
        case ActF::SIGMOID: return sigmoid(vec);
        case ActF::RELU: return relu(vec);
        default: break;
    }
    VecDouble_t v;
    return v;
}

VecDouble_t NeuralNetwork_t::relu (VecDouble_t const& vec) const{
    VecDouble_t result(vec.size(),0.0);

    for(size_t i=0; i<vec.size() ; i++){
        result[i]=relu(vec[i]);
    }

    return result;
}

VecDouble_t NeuralNetwork_t::sigmoid (VecDouble_t const& vec) const{
    VecDouble_t result(vec.size(),0.0);

    for(size_t i=0; i<vec.size() ; i++){
        result[i]=sigmoid(vec[i]);
    }

    return result;
}

VecDouble_t NeuralNetwork_t::feedforward(VecDouble_t const& x){
    //r1 = sigmoid(x*m_layers[0])
    //r2 = sigmoid(r1*m_layers[1])
    //...
    return feedforwardinlayer(x,m_layers.size()-1);
    
}


MatDouble_t
NeuralNetwork_t::createCopyLayer(MatDouble_t const& layer){
    MatDouble_t matrixCopy;

    for(VecDouble_t v : layer){
        VecDouble_t newVec(v);
        matrixCopy.push_back(newVec);
    }
    return matrixCopy;
}

vector<MatDouble_t>
NeuralNetwork_t::getm_layers(){
    vector<MatDouble_t> res;
    for(MatDouble_t layer : m_layers){
        res.push_back(createCopyLayer(layer));
    }

    return res;
}