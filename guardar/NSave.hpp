#pragma once

#include <vector>
#include <string>

class NSave
{

    private:

    public:

    void escribir(const  std::vector<  std::vector<  std::vector<double> > > v, const std::string) const;

    std::vector< std::vector< std::vector<double> > > leer(const std::string) const;

};