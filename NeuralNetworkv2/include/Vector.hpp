#pragma once

#include <vector>
#include <functional>
#include <ostream>
#include <initializer_list>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <numeric>

template <typename W>
class Vector {

    using vec = std::vector<W>;

    private: 
        vec v;

    public: 

    explicit Vector(size_t size) : v(size) {}
    Vector(const Vector& x) : v(x.v) {}
    Vector(const std::initializer_list<W>& x) : v(x) {}
    Vector(const std::vector<W>& x) : v(x) {}
    explicit Vector() : v() {}

    explicit Vector(size_t size, auto&& lambda) : v(size) {
        std::generate(v.begin(), v.end(), lambda); 
    }
    
    size_t size() const {
        return v.size();
    }

    Vector& operator=(const Vector& other) {
        if (this != &other)
            v = other.v;

        return *this;
    }

    bool operator==(const Vector& other) const {
        return v == other.v;
    }

    bool operator!=(const Vector& other) const {
        return v != other.v;
    }

    W& operator[](std::size_t idx) { 
        return v[idx]; 
    }
    
    const W& operator[](std::size_t idx) const { 
        return v[idx]; 
    }

    template <typename T>
    Vector& pow(T exp) {
        for (auto& x : v)
            x = std::pow(x, exp);

        return *this;
    }

    double mean() const {
        return std::accumulate(v.begin(), v.end(), 0.f) / v.size();
    }

    Vector& operator++() {
        for (auto& x : v)
            x++;

        return *this;
    }

    Vector& operator--() {
        for (auto& x : v)
            x--;

        return *this;
    }

    Vector& operator+=(const Vector& other){
        for (size_t i = 0; i < other.size() && i < size(); i++)
            v[i] += other[i];

        return *this;
    }

    Vector& operator-=(const Vector& other){
        for (size_t i = 0; i < other.size() && i < size(); i++)
            v[i] -= other[i];

        return *this;
    }

    Vector& operator*=(const Vector& other){
        for (size_t i = 0; i < other.size() && i < size(); i++)
            v[i] *= other[i];

        return *this;
    }

    template <typename T>
    Vector& operator+=(const T& other){
        for (size_t i = 0; i < size(); i++)
            v[i] += other;

        return *this;
    }

    template <typename T>
    Vector& operator*=(const T& other) {
        for (size_t i = 0; i < size(); i++)
            v[i] *= other;

        return *this;
    }

    template <typename T>
    Vector& operator-=(const T& other) {
        for (size_t i = 0; i < size(); i++)
            v[i] -= other;

        return *this;
    }

    Vector operator+(const Vector& other) const {
        Vector vmax(other.size() >= size() ? other : *this);
        size_t min = other.size() < size() ? other.size() : size();

        for (size_t i = 0; i < min; i++)
            vmax[i] += other.size() < size()? other[i] : v[i];

        return vmax;
    }

    Vector operator-(const Vector& other) const {
        Vector vmax(*this);

        for (size_t i = 0; i < vmax.size() && i < other.size(); i++)
            vmax[i] -= other[i];

        return vmax;
    }

    Vector operator*(const Vector& other) const {
        Vector vmax(other.size() >= size() ? other : *this);
        size_t min = other.size() < size() ? other.size() : size();

        for (size_t i = 0; i < min; i++)
            vmax[i] *= other.size() < size()? other[i] : v[i];

        return vmax;
    }

    template <typename T>
    Vector operator+(const T& num) const {
        Vector vmax(*this);

        for (size_t i = 0; i < size(); i++)
            vmax[i] += num;

        return vmax;
    }

    template <typename T>
    Vector operator*(const T& num) const {
        Vector vmax(*this);

        for (size_t i = 0; i < size(); i++)
            vmax[i] *= num;

        return vmax;
    }

    template <typename T>
    Vector operator-(const T& num) const {
        Vector vmax(*this);

        for (size_t i = 0; i < size(); i++)
            vmax[i] -= num;

        return vmax;
    }

    auto begin() const {
        return v.begin();
    }

    auto end() const {
        return v.begin();
    }

    void push_back(W e) {
        return v.push_back(e);
    }

    Vector apply(const std::function<W (W)> &func) const {
        Vector temp(*this);

        for (size_t i = 0; i < size(); i++)
            temp[i] = func(temp[i]);

        return temp;
    }

    template <typename T>
    friend Vector operator+(const T &num, const Vector &v) {
        Vector vmax(v);

        for (size_t i = 0; i < v.size(); i++)
            vmax[i] += num;

        return vmax;
    }

    template <typename T>
    friend Vector operator*(const T &num, const Vector &v) {
        Vector vmax(v);

        for (size_t i = 0; i < v.size(); i++)
            vmax[i] *= num;

        return vmax;
    }

    template <typename T>
    friend Vector operator-(const T &num, const Vector &v) {
        Vector vmax(v);

        for (size_t i = 0; i < v.size(); i++)
            vmax[i] = num - vmax[i];

        return vmax;
    }


    template <class X>
    X toMatrix() const {
        X mat(size(), 1);

        for (size_t i = 0; i < size(); i++)
            mat[i] = Vector({ v[i] });

        return mat;
    }

    friend std::ostream& operator<<(std::ostream &os, const Vector &v) {
        os << std::fixed << std::setprecision(10) << "[";

        for (size_t i = 0; i < v.size() - 1; i++)
            os << (v[i] >= 0? " " : "") << v[i] <<", ";
        
        os << (v[v.size() - 1] >= 0? " " : "") << v[v.size() - 1] << "]";

        return os;
    }
};
