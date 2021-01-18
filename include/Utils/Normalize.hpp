#pragma once

#include <fstream> 
#include <NeuralNetworkv2/Vector.hpp>
#include <NeuralNetworkv2/Matrix.hpp>

template <class W>
class Normalize
{
    using MinMaxVect = std::vector<std::tuple<W, W>>;

    private:

        MinMaxVect minmaxv;

    public:
        explicit Normalize() {};

        void fit(const Matrix<W> &m) {
            Vector min = m.min();
            Vector max = m.max();

            for (size_t i = 0; i < max.size(); i++)
                minmaxv.push_back({ min[i], max[i] });
        } 
        
        Vector<W> transform(const Vector<W> &vec) {
            std::cout << minmaxv.size() << " = " << vec.size() << std::endl;
            assert(vec.size() <= minmaxv.size());

            Vector v(vec);

            for (size_t i = 0; i < v.size(); i++) {
                auto [ min, max ] = minmaxv[i];
                v[i] = (v[i] - min) / ((max - min) + !(max - min));
            }

            return v;
        }

        Matrix<W> transform(const Matrix<W> &m) {
            Matrix<W> mat(m.size(), m.ncol);

            for (size_t i = 0; i < m.size(); i++) {
                mat[i] = transform(m[i]);
            }

            return mat;
        }

        Matrix<W> fitTransform(const Matrix<W> &m) {
            fit(m);
            return transform(m); 
        }

        auto getMinMax() const {
            return minmaxv;
        }

        void save(const std::string &fileName) {
            std::ofstream file(fileName.c_str());
            assert(file.is_open());
            file << minmaxv.size() << std::endl;
            for (auto const &minmax : minmaxv) {
                auto [ min, max ] = minmax;
                file << min << "," << max << std::endl;
            }
        }

        void load(const std::string &fileName) {
            std::ifstream file(fileName);
            std::string token;
            assert(file.is_open());

            std::getline(file, token);
            size_t size = atoi(token.c_str());

            for (size_t i = 0; i < size; i++) {
                std::getline(file, token, ',');
                double min = atof(token.c_str());
                std::getline(file, token);
                double max = atof(token.c_str());
                minmaxv.push_back({ min, max });
            }
        }
};