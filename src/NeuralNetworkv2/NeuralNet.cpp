#include <NeuralNetworkv2/NeuralNet.hpp>
#include <Utils/NSave.hpp>
#include <NeuralNetworkv2/NeuralLayer.hpp>
#include <NeuralNetworkv2/Functions.hpp>
#include <list>
#include <tuple>

NNet::VecFordward NNet::forwardPass(const Mat2d &X) const
{
    VecFordward vecA({ X });

    for (auto const& layer : nn)
    {
        auto [actf, actfD] = layer.actf;
        auto lastA = vecA.back();
        auto z = lastA * layer.w + layer.b;
        auto a = z.apply(actf);

        vecA.push_back(a);
    }

    return vecA;
}

Mat2d NNet::ttrain(
    const Mat2d &X, const Mat2d &y, 
    const CostFunc &costf, double lr
)
{

    // Forward pass
    VecFordward out = forwardPass(X);

    // Backpropagation
    std::list<Mat2d> deltas;
    auto [cost, costD] = costf;
    Mat2d lastW;

    for (int l = nn.size() - 1; l >= 0; l--)
    {
        auto [actf, actfD] = nn[l].actf;
        auto a  = out[l + 1];
        auto al = out[l];

        if (static_cast<size_t>(l) == nn.size() - 1)
            deltas.push_front(costD(a, y) ^ a.apply(actfD));
        else
            deltas.push_front(deltas.front() * lastW.transpose() ^ a.apply(actfD));

        lastW   = nn[l].w;
        nn[l].b = nn[l].b - deltas.front().mean(1) * lr;
        nn[l].w = nn[l].w - (al.transpose() * deltas.front()).mult(lr);
    }

    return out.back();
}

NNet::NNet(const std::vector<int16_t> &topology, const VecActFunc &vecAct)
{
    assert(topology.size() - 1 == vecAct.size());
    for (size_t i = 0; i < topology.size() - 1; i++)
        nn.push_back(NeuralLayer(topology[i], topology[i + 1], vecAct[i]));
}

NNet::NNet(const std::vector<int16_t> &topology, const ActFunc &actf)
{
    for (size_t i = 0; i < topology.size() - 1; i++)
        nn.push_back(NeuralLayer(topology[i], topology[i + 1], actf));
}

NNet::NNet() {}

void NNet::train(
    const Mat2d &X, const Mat2d &y, 
    const CostFunc &costf, size_t epochs, double lr, 
    const Vec2d& initialBias
)
{
    if (!initialBias.empty())
        nn.back().b = initialBias;

    for (size_t i = 0; i < epochs; i++)
    {

        Mat2d pY = ttrain(X, y, costf, lr);

        if (i % 1 == 0)
        {
            auto [cost, costD] = costf;
            auto loss = cost(pY, y);
            std::cout << "Loss: " << loss << std::endl;
        }
    }
}

void NNet::test(const Mat2d &X, const Mat2d &y) const
{
    Mat2d r = forwardPass(X).back().apply([](double n) { return (int)(n + 0.5); });
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

    ActFunc  actfRelu  { Functions::relu, Functions::reluD };
    ActFunc  actfSigm  { Functions::sigm, Functions::sigmD };
    VecActFunc  actf  {
            actfRelu,
            actfRelu,
            actfRelu,
            actfSigm
    };

    size_t i = 0;
    for (auto &layer : vecW) {
        nn.push_back(NeuralLayer(layer, actf[i]));
    }
}

void NNet::load(const std::string &fileName) {
    nn.clear();

    NSave loader(fileName);
    load(loader.read());
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
