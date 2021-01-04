#pragma once

#include <vector>
#include <list>
#include <NeuralLayer.hpp>
#include <Types.hpp>
#include <Vector.hpp>
#include <Matrix.hpp>

class NNet
{
    using NeuralNetwork = std::vector<NeuralLayer>;
    using VecFordward   = std::vector<Mat2d>;
    using VecWeights    = std::vector<std::vector<std::vector<double>>>;

private:
    NeuralNetwork nn;

    VecFordward forwardPass(const Mat2d &X) const;
    Mat2d ttrain(const Mat2d &X, const Mat2d &y, const CostFunc &costf, double lr);

public:
    explicit NNet(const std::vector<int16_t> &topology, const VecActFunc &vecAct);
    explicit NNet(const std::vector<int16_t> &topology, const ActFunc &actf);
    explicit NNet();

    void train(const Mat2d &X, const Mat2d &y, const CostFunc &costf, size_t epochs, double lr);
    void test(const Mat2d &X, const Mat2d &y) const;

    auto begin()  const;
    auto end()    const;
    size_t size() const;
    VecWeights getWeights() const;
    void load(const VecWeights&);

    NeuralLayer &operator[](std::size_t idx);
    const NeuralLayer &operator[](std::size_t idx) const;

    friend std::ostream &operator<<(std::ostream &os, const NNet &nn);
};