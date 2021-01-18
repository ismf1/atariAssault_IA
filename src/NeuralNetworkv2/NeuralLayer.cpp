#include <NeuralNetworkv2/NeuralLayer.hpp>
#include <NeuralNetworkv2/Functions.hpp>

NeuralLayer::NeuralLayer(int16_t nconn, int16_t nneur, const ActFunc &actf) {
    auto raD    = []() { return Functions::rand(-1, 1); };
    this->actf  = actf;
    this->b     = Vec2d(nneur, raD);
    this->w     = Mat2d(nconn, nneur, raD);
    this->nconn = nconn;
    this->nneur = nneur;
} 

NeuralLayer::NeuralLayer(Weights ws, const ActFunc &act) {
    b = *ws.begin();  
    ws.erase(ws.begin());   
    w     = Mat2d(ws);
    nneur = b.size();
    nconn = w.size();
    actf  = act;
}

NeuralLayer::Weights NeuralLayer::getWeights() const {
    Mat2d copy(w);

    copy.insert(0, b); 

    return copy.toSTLVector();
}

std::ostream& operator<<(std::ostream &os, const NeuralLayer &nl) {
    os << "Connections: " << nl.nconn << std::endl
        << "Neurals:" << nl.nneur << std::endl
        << "Weights: " << nl.w << std::endl
        << "Bias: " << nl.b << std::endl;

    return os;
}