#include <iostream>
#include <cmath>
#include <ale_interface.hpp>
#include <SDL/SDL.h>
#include <fstream>
#include <string>
#include <vector>
using namespace std;
using namespace ale;

// Global vars
const int maxSteps = 7500;
const char splitSymbol = ';';
int lastLives;
float totalReward;
ALEInterface alei;

vector<float> state;
vector<vector<float>> w;

vector<int> nImp = {//Muy importantes
   15, /*Posicion principal*/
   47,48,49, /*Velocidades enemigos*/
   50/*0x32*/,51,52, /*¿Velocidades enemigos?*/ 
   20/*Cambia cuando barra disparo entra en peligro*/,  

   //En duda
   21/*Cambia si hay enemigo lateral*/,
   23,24,25, /*Cambia si hay bala lateral*/     
   39, /*Tipo enemigo graficamente*/
   71,/*Cambia al pasar de ronda*/ 
   109, /*Posicion bala enemiga*/ 
   113/*0x71*/, /*Cambia periódicamente 40-80*/ 

   //No sabemos
   16,18, /*Cambia demasiado rapido*/
   32,33,34,35,36,37, /*Cambia demasiado rapido*/
   44,46, /*Cambian continuamente entre FC-FE, si mato a alguien se pone a FF*/
   42, /*¿No cambia?*/
   60, /*Cambia demasiado rapido*/
   101,102, /*Cambia demasiado rapido*/
   106, /*Cambia demasiado rapido*/
   121, /*Muy aleatorio*/
   67,68, /*Cambia demasiado rapido*/
   79,80, /*Cambia demasiado rapido*/

   //Por observar
   53,54,55,56/*0x38*/,61,62,
   65,69,70,72,74,85,87,91,92,
   104,105,
   114,119,120,123,125,126
};

//Pone nImp con todos los valores de la ram
void restartNimp(){
   nImp.resize(0);
   for(int i=0;i<128;i++){
      nImp.push_back(i);
   }
}

///////////////////////////////////////////////////////////////////////////////
/// Get info from RAM
///////////////////////////////////////////////////////////////////////////////
int getXPlayer(){
   return alei.getRAM().get(15+1);
}

//Por comprobar si falta algun byte
int getScore(){
   return alei.getRAM().get(5*16+9+1);
}

int getXEnemies(){
   return -1;
}

void cls() { std::printf("\033[2J");}

/*Direcciones
   X personaje principal: 0x0F
   Puntuacion: 0x59
*/
void showRAM(){
   const auto& RAM = alei.getRAM();

   uint8_t add = 0;
   std::printf("\033[H");
   std::printf("\nAD |  0  1  2  3  4  5  6  7  8  9  A  B  C  D  E  F");
   std::printf("\n====================================================");
   for(std::size_t i=0;i<8;i++){
      std::printf("\n%02X | ",add);
      for (std::size_t j=0;j<16; j++){
         add++;
         std::printf("%2X ", RAM.get(add));
      }
   }
   std::printf("\nScore: %2X\n", getScore());
   std::printf("X player: %2X\n", getXPlayer());
}

void showState(){
   uint8_t add = 0;
   std::printf("\033[H");
   std::printf("\nAD |  0  1  2  3  4  5  6  7  8  9  A  B  C  D  E  F");
   std::printf("\n====================================================");
   for(std::size_t i=0;i<8;i++){
      std::printf("\n%02X | ",add);
      for (std::size_t j=0;j<16; j++){
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
int hw(int iPerceptron){
    int sol=0;

    for(size_t i=0;i<w[iPerceptron].size();i++){
        sol+=w[iPerceptron][i]*state[i];
    }

    if(sol<0) return -1;
    else return 1;
}

float agentStep() {
   
   float reward = 0;
   
   if(hw(0)>0.5) reward+=alei.act(PLAYER_A_RIGHTFIRE);
   if(hw(1)>0.5) reward+=alei.act(PLAYER_A_LEFTFIRE);
   if(hw(2)>0.5) reward+=alei.act(PLAYER_A_UPFIRE);
   if(hw(3)>0.5) reward += alei.act(PLAYER_A_LEFT);
   if(hw(4)>0.5) reward += alei.act(PLAYER_A_RIGHT);
   
   return (reward + alei.act(PLAYER_A_NOOP));
}

///////////////////////////////////////////////////////////////////////////////
/// Print usage and exit
///////////////////////////////////////////////////////////////////////////////
void usage(char* pname) {
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
void readWeights(int nIA){
   int iIA=0;
   int i=0;
   int j=0;
   string line;
   char word[10];
   ifstream myfile ("IA.txt");
   if(myfile.is_open()){
      while(getline (myfile,line)){
         i=0;
         if(iIA==nIA){
            while((size_t)i!=line.size()){
               //----------------Perceptron x----------------------
               while(line[i]=='{' || line[i]==' ' || line[i]==',') i++;
               vector<float> v;
               while(line[i]!='}'){
                  j=0;
                  while(line[i]!=',' && line[i]!=' ' && line[i]!='}'){
                     word[j]=line[i];
                     j++;
                     i++;
                  }
                  if(line[i]!='}')i++;
                  word[j]='\0';

                  v.push_back(atof(word));
               }i++;
               w.push_back(v);
               //---------------------------------------------------
            }
            
         }
         iIA++;
      }
      myfile.close();
   }else{
      std::cout << "Error al abrir el archivo IA.txt" << std::endl;
   }
}

void updateState(){
   state.resize(0);
   int ramValue;

   const auto& RAM = alei.getRAM();

   state.push_back(1);  //Para multiplicar por el bias

   /*for(std::size_t i=0;i<8;i++){
      for (std::size_t j=0;j<16; j++){
         add++;
         ramValue=(int)RAM.get(add);
         state.push_back((float)ramValue);
         //printf("%2X==%2X\n",(uint8_t)ramValue,(uint8_t)state[i*16+j]);
      }
   }*/
   for(size_t i=0;i<nImp.size();i++){
      ramValue=(int)RAM.get(nImp[i]+1);
      state.push_back((float)ramValue);
   }
   
}

void saveStateInFile(){
   std::ofstream outfile;

   outfile.open("data.txt", std::ios_base::app);
   //WriteRAMValues
   const auto& RAM = alei.getRAM();

   uint8_t add = 0;
   for(std::size_t i=0;i<8;i++){
      for (std::size_t j=0;j<16; j++){
         add++;
         outfile <<  (int)RAM.get(add) << splitSymbol;
      }
   }

   //Write input values
   Uint8 *keystates = SDL_GetKeyState( NULL ); 

   outfile << (int)(keystates[SDLK_SPACE] &&
               keystates[SDLK_RIGHT]) << splitSymbol;  //Disparo derecha
   outfile << (int)(keystates[SDLK_SPACE]
               && keystates[SDLK_LEFT]) << splitSymbol;  //Disparo izquierda
   outfile << (int)(keystates[SDLK_SPACE] &&
               !keystates[SDLK_RIGHT] &&
               !keystates[SDLK_LEFT]) << splitSymbol;  //Disparo normal

   outfile << (int)keystates[SDLK_LEFT] << splitSymbol;
   outfile << (int)keystates[SDLK_RIGHT] << "\n";

}

///////////////////////////////////////////////////////////////////////////////
/// MAIN PROGRAM
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
   // Check input parameter
   if (argc != 3)
      usage(argv[0]);

   int nIA=atoi(argv[2]);  //Numero de IA a leer del archivo de pesos
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
   readWeights(nIA);
   //Muestra vector de pesos cargado
   /*for(int i=0;i<w.size();i++){
      for(int j=0;j<w[i].size();j++){
         cout << w[i][j] << ",";
      }
      cout << endl << endl;
   }*/

   // Main loop
   //alei.act(PLAYER_A_FIRE);
   int step;
   for (step = 0; 
        !alei.game_over() && step < maxSteps; 
        ++step) 
   {
      updateState();
      totalReward += agentStep();   //Movimiento
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