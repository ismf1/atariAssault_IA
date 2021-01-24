#pragma once

#include <vector>
#include <string>
#include <tuple>

class NSave
{
    
    using Data = std::vector<std::vector<std::vector<double>>>;
    private:
        std::string fileName;
    public:

    using en =  enum TypeActFunctions { SIGM, RELU };
    using Func = std::vector< en>;
    explicit NSave(const std::string &s) : fileName(s) {}
    void write(const Data &v, const Func &t) const;
    std::tuple<Data, Func> read() const;
};