#pragma once

#include <cassert>
#include <iostream>
#include <Vector.hpp>

template <typename W>
class Matrix
{

    using Mat = std::vector<Vector<W>>;

private:
    Mat mat;

public:
    size_t nrow;
    size_t ncol;

    explicit Matrix() : mat(), nrow(0), ncol(0) {}
    explicit Matrix(size_t nrow) : mat(nrow), nrow(nrow), ncol(0) {}
    explicit Matrix(size_t nrow, size_t ncol) : mat(nrow)
    {
        this->nrow = nrow;
        this->ncol = ncol;
        for (auto &row : mat)
            row = Vector<W>(ncol);
    }

    explicit Matrix(size_t nrow, size_t ncol, auto &&lambda) : mat(nrow)
    {
        this->nrow = nrow;
        this->ncol = ncol;

        for (auto &row : mat)
            row = Vector<W>(ncol, lambda);
    }

    explicit Matrix(const std::initializer_list<Vector<W>> &mat) : mat(mat)
    {
        this->nrow = this->mat.size();
        this->ncol = this->mat[0].size();
    }

    explicit Matrix(const std::vector<std::vector<W>> &mat) : mat(mat.size())
    {
        this->nrow = this->mat.size();
        this->ncol = mat[0].size();
        
        for (size_t i = 0; i < mat.size(); i++)
            this->mat[i] = Vector<W>(mat[i]);
    }
    
    Matrix(const Matrix &other)
    {
        this->nrow = other.nrow;
        this->ncol = other.ncol;
        for (auto const &row : other)
            mat.push_back(Vector(row));
    }

    auto begin() const
    {
        return mat.begin();
    }

    auto end() const
    {
        return mat.end();
    }

    size_t size() const
    {
        return mat.size();
    }

    template <typename T>
    Matrix &pow(T exp)
    {
        for (auto &x : mat)
            x.pow(exp);

        return *this;
    }

    double mean() const
    {
        double result = 0.f;

        for (auto const &row : mat)
            result += row.mean();

        return result / (double)size();
    }

    Vector<W> mean(int16_t axis)
    {
        Vector<W> v(ncol);

        if (axis == 1)
            for (size_t j = 0; j < ncol; j++)
            {
                for (size_t i = 0; i < nrow; i++)
                    v[j] += mat[i][j];
                v[j] /= nrow;
            }

        return v;
    }

    Matrix &operator=(const Matrix &other)
    {
        if (this != &other)
        {
            mat = other.mat;
            nrow = other.nrow;
            ncol = other.ncol;
        }

        return *this;
    }

    Vector<W> &operator[](std::size_t idx)
    {
        return mat[idx];
    }

    const Vector<W> &operator[](std::size_t idx) const
    {
        return mat[idx];
    }

    Matrix &operator*=(const Matrix &other)
    {
        assert(ncol == other.nrow);
        Matrix temp(nrow, other.ncol);

        for (size_t i = 0; i < nrow; i++)
        {
            for (size_t j = 0; j < other.ncol; j++)
            {
                temp[i][j] = 0;
                for (size_t k = 0; k < ncol; k++)
                {
                    temp[i][j] += mat[i][k] * other[k][j];
                }
            }
        }

        mat = temp.mat;

        return temp;
    }

    Matrix &operator+=(const Matrix &other)
    {
        assert(nrow == other.nrow && ncol == other.ncol);

        for (size_t i = 0; i < size(); i++)
            mat[i] += other[i];

        return *this;
    }

    Matrix &operator-=(const Matrix &other)
    {
        assert(nrow == other.nrow && ncol == other.ncol);

        for (size_t i = 0; i < size(); i++)
            mat[i] -= other[i];

        return *this;
    }

    Matrix operator+(const Vector<W> &other)
    {
        assert(ncol == other.size());

        Matrix temp(*this);

        for (size_t i = 0; i < size(); i++)
            temp[i] += other;

        return temp;
    }

    Matrix operator*(const Vector<W> &other)
    {
        assert(nrow == other.size());

        Matrix temp(*this);
        Matrix matV({other});

        return temp * matV;
    }

    Matrix mult(double num) const
    {
        Matrix temp(*this);

        for (size_t i = 0; i < size(); i++)
            temp[i] *= num;

        return temp;
    }

    Matrix operator+(const Matrix &other) const
    {
        assert(nrow == other.nrow && ncol == other.ncol);
        Matrix temp(*this);

        for (size_t i = 0; i < size(); i++)
            temp[i] += other[i];

        return temp;
    }

    Matrix operator^(const Matrix &other)
    {
        assert(nrow == other.nrow && ncol == other.ncol);
        Matrix temp(*this);

        for (size_t i = 0; i < size(); i++)
            temp[i] *= other[i];

        return temp;
    }

    Matrix operator-(const Matrix &other) const
    {
        assert(nrow == other.nrow && ncol == other.ncol);
        Matrix temp(*this);

        for (size_t i = 0; i < size(); i++)
            temp[i] -= other[i];

        return temp;
    }

    Matrix operator*(const Matrix &other) const
    {
        if (ncol == 1 && other.ncol == 1)
        {
            Vector a = toVector();
            Vector b = other.toVector();

            return (a * b).template toMatrix<Matrix<W>>();
        }

        assert(ncol == other.nrow);
        Matrix temp(nrow, other.ncol);

        for (size_t i = 0; i < nrow; i++)
            for (size_t k = 0; k < ncol; k++)
                for (size_t j = 0; j < other.ncol; j++)
                    temp[i][j] += mat[i][k] * other[k][j];

        return temp;
    }

    Matrix apply(const std::function<W(W)> &func) const
    {
        Matrix temp(*this);

        for (size_t i = 0; i < nrow; i++)
            temp[i] = temp[i].apply(func);

        return temp;
    }

    Vector<W> toVector() const
    {
        Vector<W> v(nrow);

        for (size_t i = 0; i < size(); i++)
            v[i] = mat[i][0];

        return v;
    }

    Matrix transpose() const
    {
        Matrix temp(ncol, nrow);

        for (size_t i = 0; i < size(); i++)
            for (size_t j = 0; j < ncol; j++)
                temp[j][i] = mat[i][j];

        return temp;
    }

    std::vector<std::vector<W>> toSTLVector() {
        std::vector<std::vector<W>> v(size());

        for (size_t i = 0; i < size(); i++) {
            v[i] = std::vector<W>(ncol);
            for (size_t j = 0; j < ncol; j++)
                v[i][j] = mat[i][j];
        }

        return v;
    }

    auto insert(size_t index, const Vector<W> &e) {
        return mat.insert(mat.begin() + index, e); 
    }

    void shape()
    {
        std::cout << "(" << nrow << ", " << ncol << ")" << std::endl;
    }

    friend std::ostream &operator<<(std::ostream &os, const Matrix &m)
    {
        os << "[" << std::endl;

        for (auto const &row : m)
            os << "\t" << row << "," << std::endl;

        os << "]";

        return os;
    }
};
