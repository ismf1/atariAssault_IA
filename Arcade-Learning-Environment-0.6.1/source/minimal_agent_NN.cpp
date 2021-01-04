#include <iostream>
#include <cmath>
#include "../src/ale_interface.hpp"
#include <SDL/SDL.h>
#include <fstream>
#include <string>
#include <vector>
#include <NeuralNet.hpp>
#include <Types.hpp>
#include <Functions.hpp>
#include <iterator>
#include <sstream>
#include <string>
#include <fstream>
#include <tuple>

#define get_elem_i_ct(i, t)                                                                \
    std::get<i>(t);                                                                        \
    static_assert(std::is_integral<decltype(i)>::value, #i " must be an integral type");   \
    static_assert(std::is_same<decltype(t), std::tuple<char, char>>::value, #t " must be a tuple");

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

using Data = std::tuple<Mat2d, Mat2d, Mat2d, Mat2d>;
using CSVFile = csv<float, float, float, float, float, float, float, float, float, float, 
                    float, float, float, float, float, float, float, float, float, float, 
                    float, float, float, float, float, float, float, float, float, float,
                    float, float, float, float, float, float, float, float, float, float, 
                    float, float, float, float, float, float, float, float, float, float, 
                    float, float, float, float, float, float, float, float, float, float, 
                    float, float, float, float>;

using namespace csvtools;

Data readCsv(std::string fileName) {
    
    Mat2d X, y, X_test, y_test;
    size_t i = 0;

    std::ifstream file(fileName.c_str());    

    for (auto row : CSVFile(file, ';')) {
        Vec2d tempX;
        tempX.push_back(std::get<0>(row));
        tempX.push_back(std::get<1>(row));
        tempX.push_back(std::get<2>(row));
        tempX.push_back(std::get<3>(row));
        tempX.push_back(std::get<4>(row));
        tempX.push_back(std::get<5>(row));
        tempX.push_back(std::get<6>(row));
        tempX.push_back(std::get<7>(row));
        tempX.push_back(std::get<8>(row));
        tempX.push_back(std::get<9>(row));
        tempX.push_back(std::get<10>(row));
        tempX.push_back(std::get<11>(row));
        tempX.push_back(std::get<12>(row));
        tempX.push_back(std::get<13>(row));
        tempX.push_back(std::get<14>(row));
        tempX.push_back(std::get<15>(row));
        tempX.push_back(std::get<16>(row));
        tempX.push_back(std::get<17>(row));
        tempX.push_back(std::get<18>(row));
        tempX.push_back(std::get<19>(row));
        tempX.push_back(std::get<20>(row));
        tempX.push_back(std::get<21>(row));
        tempX.push_back(std::get<22>(row));
        tempX.push_back(std::get<23>(row));
        tempX.push_back(std::get<24>(row));
        tempX.push_back(std::get<25>(row));
        tempX.push_back(std::get<26>(row));
        tempX.push_back(std::get<27>(row));
        tempX.push_back(std::get<28>(row));
        tempX.push_back(std::get<29>(row));
        tempX.push_back(std::get<30>(row));
        tempX.push_back(std::get<31>(row));
        tempX.push_back(std::get<32>(row));
        tempX.push_back(std::get<33>(row));
        tempX.push_back(std::get<34>(row));
        tempX.push_back(std::get<35>(row));
        tempX.push_back(std::get<36>(row));
        tempX.push_back(std::get<37>(row));
        tempX.push_back(std::get<38>(row));
        tempX.push_back(std::get<39>(row));
        tempX.push_back(std::get<40>(row));
        tempX.push_back(std::get<41>(row));
        tempX.push_back(std::get<42>(row));
        tempX.push_back(std::get<43>(row));
        tempX.push_back(std::get<44>(row));
        tempX.push_back(std::get<45>(row));
        tempX.push_back(std::get<46>(row));
        tempX.push_back(std::get<47>(row));
        tempX.push_back(std::get<48>(row));
        tempX.push_back(std::get<49>(row));
        tempX.push_back(std::get<50>(row));
        tempX.push_back(std::get<51>(row));
        tempX.push_back(std::get<52>(row));
        tempX.push_back(std::get<53>(row));
        tempX.push_back(std::get<54>(row));
        tempX.push_back(std::get<55>(row));
        tempX.push_back(std::get<56>(row));
        tempX.push_back(std::get<57>(row));
        tempX.push_back(std::get<58>(row));
        tempX.push_back(std::get<59>(row));        
        Vec2d tempY;
        tempY.push_back(std::get<60>(row));
        tempY.push_back(std::get<61>(row));
        tempY.push_back(std::get<62>(row));
        tempY.push_back(std::get<63>(row));
        
         X.push_back(tempX);
         y.push_back(tempY);
        i++;
    }

    X.ncol = 60;
    X.nrow = X.size();
    y.ncol = 4;
    y.nrow = y.size();
    X_test.ncol = 60;
    X_test.nrow = X_test.size();
    y_test.ncol = 4;
    y_test.nrow = y_test.size();

    return { X, y, X_test, y_test};
}

using namespace std;

// Global vars
const int maxSteps = 7500;
const char splitSymbol = ';';
int lastLives;
float totalReward;
ALEInterface alei;

vector<double> state;
vector<vector<double>> w;

vector<int> nImp = {     //Muy importantes
    15,                  /*Posicion principal*/
    47, 48, 49,          /*Velocidades enemigos*/
    50 /*0x32*/, 51, 52, /*¿Velocidades enemigos?*/
    20 /*Cambia cuando barra disparo entra en peligro*/,

    //En duda
    21 /*Cambia si hay enemigo lateral*/,
    23, 24, 25,   /*Cambia si hay bala lateral*/
    39,           /*Tipo enemigo graficamente*/
    71,           /*Cambia al pasar de ronda*/
    109,          /*Posicion bala enemiga*/
    113 /*0x71*/, /*Cambia periódicamente 40-80*/

    //No sabemos
    16, 18,                 /*Cambia demasiado rapido*/
    32, 33, 34, 35, 36, 37, /*Cambia demasiado rapido*/
    44, 46,                 /*Cambian continuamente entre FC-FE, si mato a alguien se pone a FF*/
    42,                     /*¿No cambia?*/
    60,                     /*Cambia demasiado rapido*/
    101, 102,               /*Cambia demasiado rapido*/
    106,                    /*Cambia demasiado rapido*/
    121,                    /*Muy aleatorio*/
    67, 68,                 /*Cambia demasiado rapido*/
    79, 80,                 /*Cambia demasiado rapido*/

    //Por observar
    53, 54, 55, 56 /*0x38*/, 61, 62,
    65, 69, 70, 72, 74, 85, 87, 91, 92,
    104, 105,
    114, 119, 120, 123, 125, 126};

//Pone nImp con todos los valores de la ram
void restartNimp()
{
   nImp.resize(0);
   for (int i = 0; i < 128; i++)
   {
      nImp.push_back(i);
   }
}

///////////////////////////////////////////////////////////////////////////////
/// Get info from RAM
///////////////////////////////////////////////////////////////////////////////
int getXPlayer()
{
   return alei.getRAM().get(15 + 1);
}

//Por comprobar si falta algun byte
int getScore()
{
   return alei.getRAM().get(5 * 16 + 9 + 1);
}

int getXEnemies()
{
   return -1;
}

void cls() { std::printf("\033[2J"); }

/*Direcciones
   X personaje principal: 0x0F
   Puntuacion: 0x59
*/
void showRAM()
{
   const auto &RAM = alei.getRAM();

   uint8_t add = 0;
   std::printf("\033[H");
   std::printf("\nAD |  0  1  2  3  4  5  6  7  8  9  A  B  C  D  E  F");
   std::printf("\n====================================================");
   for (std::size_t i = 0; i < 8; i++)
   {
      std::printf("\n%02X | ", add);
      for (std::size_t j = 0; j < 16; j++)
      {
         add++;
         std::printf("%2X ", RAM.get(add));
      }
   }
   std::printf("\nScore: %2X\n", getScore());
   std::printf("X player: %2X\n", getXPlayer());
}

void showState()
{
   uint8_t add = 0;
   std::printf("\033[H");
   std::printf("\nAD |  0  1  2  3  4  5  6  7  8  9  A  B  C  D  E  F");
   std::printf("\n====================================================");
   for (std::size_t i = 0; i < 8; i++)
   {
      std::printf("\n%02X | ", add);
      for (std::size_t j = 0; j < 16; j++)
      {
         add++;
         std::printf("%2X ", (uint8_t)state[add]);
      }
   }
   std::printf("\nScore: %2X\n", getScore());
   std::printf("X player: %2X\n", getXPlayer());
}

///////////////////////////////////////////////////////////////////////////////
/// Do Next Agent Step
///////////////////////////////////////////////////////////////////////////////
//Prediccion del perceptron iPerceptron con la entrada X
int hw(vector<double> X, int iPerceptron)
{
   int sol = 0;

   for (int i = 0; i < w[iPerceptron].size(); i++)
   {
      sol += w[iPerceptron][i] * state[i];
   }

   if (sol < 0)
      return -1;
   else
      return 1;
}

float agentStep(const NNet &nn)
{

   static int wide = 9;
   float reward = 0;

   std::vector<std::vector<double>> xx;
   xx.push_back(state);
   std::vector<double> result = nn.predict(Mat2d(xx));

   if (result[0] > 0)
      reward += alei.act(PLAYER_A_RIGHTFIRE);
   if (result[1] > 0)
      reward += alei.act(PLAYER_A_LEFTFIRE);
   if (result[2] > 0)
      reward += alei.act(PLAYER_A_UPFIRE);
   if (result[3] > 0)
      reward += alei.act(PLAYER_A_LEFT);
   if (result[4] > 0)
      reward += alei.act(PLAYER_A_RIGHT);

   return (reward + alei.act(PLAYER_A_NOOP));
}

///////////////////////////////////////////////////////////////////////////////
/// Print usage and exit
///////////////////////////////////////////////////////////////////////////////
void usage(char *pname)
{
   std::cerr
       << "\nUSAGE:\n"
       << "   " << pname << " <romfile> <n_IA>\n";
   exit(-1);
}

///////////////////////////////////////////////////////////////////////////////
/// Print RAM
///////////////////////////////////////////////////////////////////////////////
/*void printRAM(){  //Solo funciona bien para breakout.bin
      std::cout << "X player: " << (int)alei.getRAM().get(72) << std::endl;
      std::cout << "X ball: " << (int)alei.getRAM().get(99) << std::endl;
}*/

///////////////////////////////////////////////////////////////////////////////
/// Save State
///////////////////////////////////////////////////////////////////////////////
void readWeights(int nIA)
{
   int iIA = 0;
   int i = 0;
   int j = 0;
   string line;
   char word[10];
   ifstream myfile("IA.txt");
   if (myfile.is_open())
   {
      while (getline(myfile, line))
      {
         i = 0;
         if (iIA == nIA)
         {
            while (i != line.size())
            {
               //----------------Perceptron x----------------------
               while (line[i] == '{' || line[i] == ' ' || line[i] == ',')
                  i++;
               vector<double> v;
               while (line[i] != '}')
               {
                  j = 0;
                  while (line[i] != ',' && line[i] != ' ' && line[i] != '}')
                  {
                     word[j] = line[i];
                     j++;
                     i++;
                  }
                  if (line[i] != '}')
                     i++;
                  word[j] = '\0';

                  v.push_back(atof(word));
               }
               i++;
               w.push_back(v);
               //---------------------------------------------------
            }
         }
         iIA++;
      }
      myfile.close();
   }
   else
   {
      std::cout << "Error al abrir el archivo IA.txt" << std::endl;
   }
}

void updateState()
{
   state.resize(0);
   int ramValue;

   uint8_t add = 0;
   const auto &RAM = alei.getRAM();

   state.push_back(1); //Para multiplicar por el bias

   /*for(std::size_t i=0;i<8;i++){
      for (std::size_t j=0;j<16; j++){
         add++;
         ramValue=(int)RAM.get(add);
         state.push_back((float)ramValue);
         //printf("%2X==%2X\n",(uint8_t)ramValue,(uint8_t)state[i*16+j]);
      }
   }*/
   for (int i = 0; i < nImp.size(); i++)
   {
      ramValue = (int)RAM.get(nImp[i] + 1);
      state.push_back((float)ramValue);
   }
}

void saveStateInFile()
{
   std::ofstream outfile;

   outfile.open("data.txt", std::ios_base::app);
   //WriteRAMValues
   const auto &RAM = alei.getRAM();

   uint8_t add = 0;
   for (std::size_t i = 0; i < 8; i++)
   {
      for (std::size_t j = 0; j < 16; j++)
      {
         add++;
         outfile << (int)RAM.get(add) << splitSymbol;
      }
   }

   //Write input values
   Uint8 *keystates = SDL_GetKeyState(NULL);

   outfile << (int)(keystates[SDLK_SPACE] &&
                    keystates[SDLK_RIGHT])
           << splitSymbol;                                                         //Disparo derecha
   outfile << (int)(keystates[SDLK_SPACE] && keystates[SDLK_LEFT]) << splitSymbol; //Disparo izquierda
   outfile << (int)(keystates[SDLK_SPACE] &&
                    !keystates[SDLK_RIGHT] &&
                    !keystates[SDLK_LEFT])
           << splitSymbol; //Disparo normal

   outfile << (int)keystates[SDLK_LEFT] << splitSymbol;
   outfile << (int)keystates[SDLK_RIGHT] << "\n";
}

///////////////////////////////////////////////////////////////////////////////
/// MAIN PROGRAM
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
   // Check input parameter
   if (argc != 3)
      usage(argv[0]);

   int nIA = atoi(argv[2]); //Numero de IA a leer del archivo de pesos
   // Create alei object.
   alei.setInt("random_seed", 0);
   alei.setFloat("repeat_action_probability", 0);
   alei.setBool("display_screen", true);
   alei.setBool("sound", true);
   alei.loadROM(argv[1]);

   // Init
   srand(time(NULL));
   lastLives = alei.lives();
   totalReward = .0f;
   //restartNimp(); //Usa toda la RAM
   // readWeights(nIA);
   //Muestra vector de pesos cargado
   /*for(int i=0;i<w.size();i++){
      for(int j=0;j<w[i].size();j++){
         cout << w[i][j] << ",";
      }
      cout << endl << endl;
   }*/

   // Main loop
   //alei.act(PLAYER_A_FIRE);
   int16_t p = 60;
   std::vector<int16_t> topology = {p, 128, 64, 32, 16, 4};
   CostFunc costf{Functions::mse, Functions::mseD};
   ActFunc actfRelu{Functions::relu, Functions::reluD};
   ActFunc actfSigm{Functions::sigm, Functions::sigmD};
   VecActFunc actf {
       actfSigm,
       actfRelu,
       actfRelu,
       actfRelu,
       actfSigm
   };
   auto [X, y, X_test, y_test] = readCsv("dataBuena.txt");

   // std::cout << X << std::endl;
   // std::cout << y << std::endl;

   NNet nn(topology, actf);
   nn.train(X, y, costf, 50, 5e-6f);

   int step;
   for (step = 0;
        !alei.game_over() && step < maxSteps;
        ++step)
   {
      updateState();
      totalReward += agentStep(nn); //Movimiento
      cls();
      //showRAM();
      //cout << endl;
      showState();

      //saveStateInFile();
   }

   std::cout << "Steps: " << step << std::endl;
   std::cout << "Reward: " << totalReward << std::endl;

   return 0;
}