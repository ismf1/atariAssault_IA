#include <NeuralNetworkv2/NeuralNet.hpp>
#include <NeuralNetworkv2/Functions.hpp>
#include <iterator>
#include <sstream>
#include <string>
#include <fstream>
#include <tuple>
#include <Utils/NSave.hpp>
#include <Utils/Data.hpp>

void print(std::vector<std::vector<std::vector<double>>> v) {
    for (auto &i : v) {
        for (auto &j : i) {
            for (auto &e : j) {
                std::cout << e  << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

namespace csvtools {
    /// Read the last element of the tuple without calling recursively
    template <std::size_t idx, class... fields>
    typename std::enable_if<idx >= std::tuple_size<std::tuple<fields...>>::value - 1>::type
    read_tuple(std::istream &in, std::tuple<fields...> &out, const char delimiter) {
        std::string cell;
        std::getline(in, cell, delimiter);
        std::stringstream cell_stream(cell);
        cell_stream >> std::get<idx>(out);
    }

    /// Read the @p idx-th element of the tuple and then calls itself with @p idx + 1 to
    /// read the next element of the tuple. Automatically falls in the previous case when
    /// reaches the last element of the tuple thanks to enable_if
    template <std::size_t idx, class... fields>
    typename std::enable_if<idx < std::tuple_size<std::tuple<fields...>>::value - 1>::type
    read_tuple(std::istream &in, std::tuple<fields...> &out, const char delimiter) {
        std::string cell;
        std::getline(in, cell, delimiter);
        std::stringstream cell_stream(cell);
        cell_stream >> std::get<idx>(out);
        read_tuple<idx + 1, fields...>(in, out, delimiter);
    }
}

/// Iterable csv wrapper around a stream. @p fields the list of types that form up a row.
template <class... fields>
class csv {
    std::istream &_in;
    const char _delim;
public:
    typedef std::tuple<fields...> value_type;
    class iterator;

    /// Construct from a stream.
    inline csv(std::istream &in, const char delim) : _in(in), _delim(delim) {}

    /// Status of the underlying stream
    /// @{
    inline bool good() const {
        return _in.good();
    }
    inline const std::istream &underlying_stream() const {
        return _in;
    }
    /// @}

    inline iterator begin();
    inline iterator end();
private:

    /// Reads a line into a stringstream, and then reads the line into a tuple, that is returned
    inline value_type read_row() {
        std::string line;
        std::getline(_in, line);
        std::stringstream line_stream(line);
        std::tuple<fields...> retval;
        csvtools::read_tuple<0, fields...>(line_stream, retval, _delim);
        return retval;
    }
};

/// Iterator; just calls recursively @ref csv::read_row and stores the result.
template <class... fields>
class csv<fields...>::iterator {
    csv::value_type _row;
    csv *_parent;
public:
    typedef std::input_iterator_tag iterator_category;
    typedef csv::value_type         value_type;
    typedef std::size_t             difference_type;
    typedef csv::value_type *       pointer;
    typedef csv::value_type &       reference;

    /// Construct an empty/end iterator
    inline iterator() : _parent(nullptr) {}
    /// Construct an iterator at the beginning of the @p parent csv object.
    inline iterator(csv &parent) : _parent(parent.good() ? &parent : nullptr) {
        ++(*this);
    }

    /// Read one row, if possible. Set to end if parent is not good anymore.
    inline iterator &operator++() {
        if (_parent != nullptr) {
            _row = _parent->read_row();
            if (!_parent->good()) {
                _parent = nullptr;
            }
        }
        return *this;
    }

    inline iterator operator++(int) {
        iterator copy = *this;
        ++(*this);
        return copy;
    }

    inline csv::value_type const &operator*() const {
        return _row;
    }

    inline csv::value_type const *operator->() const {
        return &_row;
    }

    bool operator==(iterator const &other) {
        return (this == &other) or (_parent == nullptr and other._parent == nullptr);
    }
    bool operator!=(iterator const &other) {
        return not (*this == other);
    }
};

template <class... fields>
typename csv<fields...>::iterator csv<fields...>::begin() {
    return iterator(*this);
}

template <class... fields>
typename csv<fields...>::iterator csv<fields...>::end() {
    return iterator();
}

int main(int argc, char *argv[]) {

        std::cout << "Escalando..." << std::endl;
    if (argc < 3) {
        std::cerr << "ERROR: Params" << std::endl
                  << "<program> -t dataFile outModelFile" << std::endl
                  << "<program> -l modelFile" << std::endl;
    }

    std::string opt  = argv[1];
    std::string file = argv[2];

    if (opt == "-t") {

        std::string fileModel = argv[3];
        Data data;

        data.init(file.c_str(), 59, 5);

        auto X = Mat2d(data.X);
        auto y = Mat2d(data.Y);

        std::cout << "Escalando..." << std::endl;
        Scaler2d scaler;
        X = scaler.fitTransform(X);
        scaler.save("escaladorSuperBonito.txt");
        std::cout << "Ya puedo entrenar mi super red" << std::endl;

        std::vector<int16_t> topology = { (int16_t)X.ncol, 128, 64, (int16_t)y.ncol };
        CostFunc costf     { Functions::mse, Functions::mseD   };
        ActFunc  actfRelu  { Functions::relu, Functions::reluD };
        ActFunc  actfSigm  { Functions::sigm, Functions::sigmD };
        VecActFunc  actf  {
            actfRelu,
            actfRelu,
            actfSigm
        };
        
        Vec2d initialBias(y.ncol);
        Vec2d classWeights(y.ncol);

        for (size_t i = 0; i < y.ncol; i++) {
            double pos = y.countIf(i, [](double e) { return e == 1; }); 
            double neg = y.countIf(i, [](double e) { return e == 0; });
            std::cout << "Positive: " << pos << ", Negative: " << neg << std::endl;
            initialBias[i] = std::log(pos / neg);
            classWeights[i] = (1 / pos) * y.size() / 2;
        }

        NNet nn(topology, actf);
        NSave saver(fileModel);
        // std::cout << nn << std::endl;
        std::cout << "Bias inicial" << std::endl;
        std::cout << initialBias << std::endl;
        X.shape();
        std::cout << "Pesos" << std::endl;
        std::cout << classWeights << std::endl;

        nn.train(X, y, costf, atof(argv[4]), atof(argv[5]), initialBias);
        // nn.train(X, y, costf, atof(argv[4]), atof(argv[5]));
        // nn.test(X_test, y_test);
        // saver.write(nn.getWeights());
    } 
    else if (opt == "-l") {
        NNet nn;
        NSave saver(file);
        // nn.load(saver.read());
    } else {
        std::cerr << "ERROR: Params" << std::endl
            << "<program> -t dataFile outModelFile" << std::endl
            << "<program> -l modelFile" << std::endl;
    }

    return 0;
}