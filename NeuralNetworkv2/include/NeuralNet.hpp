#pragma once

#include <vector>
#include <list>
#include <iostream>
#include <NeuralLayer.hpp>
#include <Types.hpp>
#include <Vector.hpp>
#include <Matrix.hpp>
#include <Network.hpp>

class NNet : public Network
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

    void train(const Mat2d &X, const Mat2d &y, const CostFunc &costf, size_t epochs, double lr=0.1f, const Vec2d& initialBias=Vec2d());
    void test(const Mat2d &X, const Mat2d &y) const;

    std::vector<double> predict(const std::vector<double> &X) {
        std::vector<std::vector<double>> a; 
        a.push_back(X);
        Mat2d temp(a);
        std::cout << temp << std::endl;
        Mat2d r = forwardPass(temp).back();
        return r.toSTLVector()[0];
    };

    auto begin()  const;
    auto end()    const;
    size_t size() const;
    VecWeights getWeights() const;
    void load(const VecWeights&);
    void load(const std::string&);

    NeuralLayer &operator[](std::size_t idx);
    const NeuralLayer &operator[](std::size_t idx) const;

    friend std::ostream &operator<<(std::ostream &os, const NNet &nn);
};