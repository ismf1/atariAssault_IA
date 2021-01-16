#pragma once

#include <vector>
#include <string>

class NSave
{
    using Data = std::vector<std::vector<std::vector<double>>>;

    private:
        std::string fileName;

    public:

    explicit NSave(const std::string &s) : fileName(s) {}
    void write(const Data &v) const;
    Data read() const;
};