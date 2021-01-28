#ifndef BALANCE_H
#define BALANCE_H
#include <stdio.h>
#include <vector>
#include <tuple>

using Vec_d = std::vector<double>;
using Mat_d = std::vector<Vec_d>;
using Vec_st = std::vector<size_t>;

class Balance{

    private:
    size_t col1;
    size_t col2;
    size_t col3;
    size_t col4;
    size_t col5;

    public:
    Balance(const Vec_st &v);
    Vec_d calculaVecino(Vec_d const& vecino, Vec_d const& original);
    Vec_d generateVecino(Mat_d const& m, size_t const& i);
    std::tuple<Mat_d, Mat_d> generate(Mat_d const& X, Mat_d const& y);
    
        
};

#endif
