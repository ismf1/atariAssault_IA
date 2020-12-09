#include <vector>
#include <initializer_list>
#include <stdint.h>
#include <stdexcept>
#include <random>

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

        VecDouble_t result(x);
        for (auto const& Wi : m_layers){
            //Capa Wi
            //Añadimos el x0 = 1
            result.resize(result.size()+1); //Aumentamos size
            copy(result.rbegin()+1, result.rend(), result.rbegin()); //Desplazamos los elementos 1 pos a la derecha
            result[0]=1.0;

            result = sigmoid(multiplyT(result,Wi));
        }

        return result;
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

void randomTrain(initializer_list<uint16_t> layerStruct, NeuralNetwork_t& bestNet,MatDouble_t const& X, VecDouble_t const& Y){
    double lessFails=pow(X.size(),2);
    double fails=0;


    for(size_t i=0;i<5000;i++){
        //Iteracion i
        NeuralNetwork_t net(layerStruct);
        int j=0;
        fails=0;
        for (auto const& xi : X){
            //Probamos con la entrada xi (iterador j)
            auto res = net.feedforward(xi);

            fails += pow(Y[j]-res[0],2);
            
            j++;
        }
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
    
    randomTrain(layerStruct,net,X,Y);

    //Predecimos los valores
    auto res = net.feedforward({0.0 , 0.0});
    printf("0.0 xor 0.0 = %i\n",(int)(res[0]+0.5));
    res = net.feedforward({0.0 , 1.0});
    printf("0.0 xor 1.0 = %i\n",(int)(res[0]+0.5));
    res = net.feedforward({1.0 , 0.0});
    printf("1.0 xor 0.0 = %i\n",(int)(res[0]+0.5));
    res = net.feedforward({1.0 , 1.0});
    printf("1.0 xor 1.0 = %i\n",(int)(res[0]+0.5));
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