#include <NeuralNet.hpp>
#include <list>

NNet::VecFordward NNet::forwardPass(const Mat2d &X) const
{
    VecFordward vecA;

    vecA.push_back({ Mat2d(), X });

    for (auto const& layer : nn)
    {
        auto [actf, actfD] = layer.actf;
        auto [lastZ, lastA] = vecA.back();
        auto z = lastA * layer.w + layer.b;
        auto a = z.apply(actf);

        vecA.push_back({ z, a });
    }

    return vecA;
}

Mat2d NNet::ttrain(const Mat2d &X, const Mat2d &y, const CostFunc &costf, double lr)
{

    // Forward pass
    VecFordward out = forwardPass(X);

    // Backward pass
    std::list<Mat2d> deltas;
    auto [cost, costD] = costf;
    Mat2d _W;

    for (int l = nn.size() - 1; l >= 0; l--)
    {
        auto [actf, actfD] = nn[l].actf;
        auto [z, a]  = out[l + 1];
        auto [zl, al] = out[l];

        if (static_cast<size_t>(l) == nn.size() - 1)
            deltas.push_front(costD(a, y) ^ a.apply(actfD));
        else
            deltas.push_front(deltas.front() * _W.transpose() ^ a.apply(actfD));

        _W      = nn[l].w;
        nn[l].b = nn[l].b - deltas.front().mean(1) * lr;
        nn[l].w = nn[l].w - (al.transpose() * deltas.front()).mult(lr);
    }

    auto [Z, A] = out.back();
    return A;
}

NNet::NNet(const std::vector<int16_t> &topology, const VecActFunc &vecAct)
{
    for (size_t i = 0; i < topology.size() - 1; i++) {
        std::cout << i << std::endl;
        nn.push_back(NeuralLayer(topology[i], topology[i + 1], vecAct[i]));
    }
}

NNet::NNet(const std::vector<int16_t> &topology, const ActFunc &actf)
{
    for (size_t i = 0; i < topology.size() - 1; i++)
        nn.push_back(NeuralLayer(topology[i], topology[i + 1], actf));
}

NNet::NNet() {}

void NNet::train(const Mat2d &X, const Mat2d &y, const CostFunc &costf, size_t epochs, double lr)
{

    for (size_t i = 0; i < epochs; i++)
    {

        Mat2d pY = ttrain(X, y, costf, lr);

        if (i % 1 == 0)
        {
            auto [cost, costD] = costf;
            auto loss = cost(pY, y);
            std::cout << "MSE: " << loss << std::endl;
        }
    }
}

void NNet::test(const Mat2d &X, const Mat2d &y) const
{
    auto [z, a] = forwardPass(X).back();
    Mat2d r = a.apply([](double n) { return (int)(n + 0.5); });
    double acc = 0;

    for (size_t i = 0; i < r.size(); i++) {
        std::cout << y[i]  << "==" << r[i] << std::endl; 
        if (y[i] == r[i])
            acc++;
    }

    std::cout << "Acc: " << (acc * 100.f / r.size()) << "%" << std::endl;
}

NNet::VecWeights NNet::getWeights() const {
    VecWeights vecW;

    for (size_t i = 0; i < size(); i++)
        vecW.push_back(nn[i].getWeights());

    return vecW;
}

void NNet::load(const NNet::VecWeights &vecW) {
    nn.clear();

    for (auto &layer : vecW)
        nn.push_back(NeuralLayer(layer));
}

auto NNet::begin() const
{
    return nn.begin();
}

auto NNet::end() const
{
    return nn.end();
}

size_t NNet::size() const
{
    return nn.size();
}

NeuralLayer &NNet::operator[](std::size_t idx)
{
    return nn[idx];
}

const NeuralLayer &NNet::operator[](std::size_t idx) const
{
    return nn[idx];
}

std::ostream &operator<<(std::ostream &os, const NNet &nn)
{
    for (size_t i = 0; i < nn.size(); i++)
    {
        os << "---Neural Layer " << i << "---" << std::endl
           << nn[i] << std::endl;
    }

    return os;
}
