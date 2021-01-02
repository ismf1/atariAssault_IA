#include <NeuralNet.hpp>
#include <list>

NNet::VecFordward NNet::forwardPass(const Mat2d &X) const
{
    VecFordward out;

    out.push_back(X);

    for (size_t i = 0; i < nn.size(); i++)
    {
        auto [actf, actfD] = nn[i].actf;
        auto lastA = out.back();
        auto z = lastA * nn[i].w + nn[i].b;
        auto a = z.apply(actf);
        out.push_back(a);
    }

    return out;
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
        auto a = out[l + 1];
        auto al = out[l];

        if (static_cast<size_t>(l) == nn.size() - 1)
            deltas.push_front(costD(a, y) * a.apply(actfD));
        else
            deltas.push_front(deltas.front() * _W.transpose() ^ a.apply(actfD));

        _W = nn[l].w;
        nn[l].b = nn[l].b - deltas.front().mean(1) * lr;
        nn[l].w = nn[l].w - (al.transpose() * deltas.front()).mult(lr);
    }

    return out.back();
}

NNet::NNet(const std::vector<int16_t> &topology, const VecActFunc &vecAct)
{
    for (size_t i = 0; i < topology.size() - 1; i++)
        nn.push_back(NeuralLayer(topology[i], topology[i + 1], vecAct[i]));
}

NNet::NNet(const std::vector<int16_t> &topology, const ActFunc &actf)
{
    for (size_t i = 0; i < topology.size() - 1; i++)
        nn.push_back(NeuralLayer(topology[i], topology[i + 1], actf));
}

void NNet::train(const Mat2d &X, const Mat2d &y, const CostFunc &costf, size_t epochs, double lr)
{

    for (size_t i = 0; i < epochs; i++)
    {

        Mat2d pY = ttrain(X, y, costf, lr);

        if (i % 25 == 0)
        {
            auto [cost, costD] = costf;
            auto loss = cost(pY, y);
            std::cout << "MSE: " << loss << std::endl;
        }
    }
}

void NNet::test(const Mat2d &X, const Mat2d &y) const
{
    double acc = 0;
    double err = 0;

    Mat2d r = forwardPass(X).back().apply([](double n) { return (int)(n + 0.5); });

    for (size_t i = 0; i < r.size(); i++)
    {
        std::cout << y[i] << "==" << r[i] << std::endl;
        if (y[i] == r[i])
            acc++;
        else
            err++;
    }

    std::cout << "Resultado de testing: " << (acc * 100.f / r.size()) << "%" << std::endl;
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
