#pragma once

#include <iostream>
#include <vector>

class Network
{
    using Data = std::vector<double>;

    public:
        virtual void load(const std::string &) = 0;
        virtual Data predict(const Data &)     = 0;
};