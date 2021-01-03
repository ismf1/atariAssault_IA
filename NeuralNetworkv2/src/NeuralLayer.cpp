#include <NeuralLayer.hpp>
#include <Functions.hpp>

NeuralLayer::NeuralLayer(int16_t nconn, int16_t nneur, ActFunc actf) {
    auto raD    = []() { return Functions::rand(-1, 1); };
    this->actf  = actf;
    this->b     = Vec2d(nneur, raD);
    this->w     = Mat2d(nconn, nneur, raD);
    this->nconn = nconn;
    this->nneur = nneur;
} 

std::ostream& operator<<(std::ostream &os, const NeuralLayer &nl) {
    os << "Connections: " << nl.nconn << std::endl
        << "Neurals:" << nl.nneur << std::endl
        << "Weights: " << nl.w;

    return os;
}

std::ostream& NeuralLayer::toStream(std::ostream &os) const {

    std::ostream a(w.toStream(os));

    os << nconn << std::endl
       << nneur << std::endl
       <<  << std::endl;

}