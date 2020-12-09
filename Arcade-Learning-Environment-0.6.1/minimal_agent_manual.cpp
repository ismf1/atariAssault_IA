#include <iostream>
#include <cmath>
#include "src/ale_interface.hpp"
#include <SDL/SDL.h>
#include <fstream>
#include <vector>
using namespace std;

// Global vars
const int maxSteps = 7500;
const char splitSymbol = ';';
int lastLives;
float totalReward;
ALEInterface alei;
vector<int> currentState;
vector<int> changes;
bool gameOverManual=false;

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

void initVector(vector<int> &v){
   for(std::size_t i=0;i<8;i++){
      for (std::size_t j=0;j<16; j++){
         v.push_back(0);
      }
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
   Cambia al matar: 0x59 = 89
   Puntuacion: 0x00,0x01 ; 0x06,0x07
   Disparo reciente: 0x1A [0-1]
   0x73 : Solo cambia entre 9A-AC cuando no realizo accion
   Nº de rondas superadas: 0x67
   Velocidades enemigos: 0x2F-0x31 , ¿3x32-3x34?
   0x08: cambia cuando me matan entre 0x40-0x80
   0x0A: cambia al pasar de ronda entre 0x60-0x70
   0x09: cambia cuando me matan
   Posicion de mi bala vertical: 0x42
   Posicion balas laterales: 0x11
   0x17-0x19: Cambia al disparar bala lateral
   0x16: Cambia al morir
   0x15: Cambia si hay enemigo lateral
   Barra disparo: 0x1B 0x1C , al llenarse mueres
   Barra de disparo en peligro: 0x14
   0x1D,0x1E: Cambian cuando muero
   Tipo enemigo graficamente: 0x27
   0x48 : Cambia al pasar de ronda
   Vidas restantes: 0x64
   0x6B : Cambia a 0x33 al pasar de ronda y vuelve a 0
   0x6E : Cambia al cambiar de ronda (a veces)
   0x6D : Posicion bala enemiga
   0x26 : Cambia al disparar de forma no seguida
   0x2B : Sprite enemigo de abajo
   0x2D : Cambia al matar al enemigo de abajo
   0x3B : Cambia al comenzar y al disparar
   0x7A : Velocidad nave grande enemiga (inofensiva)
   0x71 : Cambia periodicamente entre 0x40-0x80

   Cambian mas de 2 veces:
                           {0,1,2,6,7,8,9,10,15,16,17,18,
                           21,22,23,24,25,26,27,29,30,32,33,
                           34,35,36,37,38,39,42,43,44,45,46,47
                           48,49,50,51,52,53,54,55,56,59,60,61
                           62,65,66,67,68,69,70,71,72,74,79,
                           80,85,87,89,91,92,100,101,102,103,
                           104,105,106,107,109,110,113,114,
                           115,119,120,121,122,123,125,126,} 

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
         if((int)RAM.get(add)!=currentState[add-1]){
            cout <<"\033[31m";
         }
         std::printf("%2X ", RAM.get(add));
         cout <<"\033[0m";
      }
   }
   std::printf("\nScore: %2X\n", getScore());
   std::printf("X player: %2X\n", getXPlayer());
}

void compareRAM(){
   const auto& RAM = alei.getRAM();

   uint8_t add = 0;

   for(std::size_t i=0;i<8;i++){
      for (std::size_t j=0;j<16; j++){
         add++;
         if((int)RAM.get(add)!=currentState[i*16+j]){
            changes[i*16+j]++;
         }
         currentState[i*16+j]=(int)RAM.get(add);
      }
   }
   std::printf("\nScore: %2X\n", getScore());
   std::printf("X player: %2X\n", getXPlayer());
}

void showChanges(){
   uint8_t add = 0;
   vector<uint8_t> moreThan1;

   std::printf("\nAD |  0  1  2  3  4  5  6  7  8  9  A  B  C  D  E  F");
   std::printf("\n====================================================");
   for(std::size_t i=0;i<8;i++){
      std::printf("\n%02X | ",add);
      for (std::size_t j=0;j<16; j++){
         std::printf("%2X ", changes[add]);
         if(changes[add]>2 && changes[add]<10) moreThan1.push_back(add);
         add++;
      }
   }

   /*for(int i=0;i<moreThan1.size();i++){
      cout << (int)moreThan1[i] << " , " << endl;
   }*/

   cout << "\nNumero de posiciones importantes: " << nImp.size() << '/' << currentState.size() << endl;


}

///////////////////////////////////////////////////////////////////////////////
/// Do Next Agent Step
///////////////////////////////////////////////////////////////////////////////
float agentStep() {
   
   static int wide = 9;
   float reward = 0;

   //Get keys state
   Uint8 *keystates = SDL_GetKeyState( NULL ); 

   if(keystates[SDLK_q])gameOverManual=true;

   //if (alei.lives() != lastLives) {
   if(keystates[SDLK_SPACE]){ //Si pulsamos espacio
      if(keystates[SDLK_RIGHT]){
         reward+=alei.act(PLAYER_A_RIGHTFIRE);
      }else if(keystates[SDLK_LEFT]){
         reward+=alei.act(PLAYER_A_LEFTFIRE);
      }else{
         //--lastLives;
         //reward+=alei.act(PLAYER_A_FIRE);
         reward+=alei.act(PLAYER_A_UPFIRE);
      }
   }

   // Apply rules.

   if (keystates[SDLK_LEFT]) { //Si pulsamos izquierda
      reward += alei.act(PLAYER_A_LEFT);
   }
   if (keystates[SDLK_RIGHT]) {   //Si pulsamos derecha
      reward += alei.act(PLAYER_A_RIGHT);
   } 
   
   return (reward + alei.act(PLAYER_A_NOOP));
}

///////////////////////////////////////////////////////////////////////////////
/// Print usage and exit
///////////////////////////////////////////////////////////////////////////////
void usage(char* pname) {
   std::cerr
      << "\nUSAGE:\n" 
      << "   " << pname << " <romfile>\n";
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

void saveStateInFile(){
   std::ofstream outfile;

   outfile.open("data.txt", std::ios_base::app);
   //WriteRAMValues
   const auto& RAM = alei.getRAM();

   /*uint8_t add = 0;
   for(std::size_t i=0;i<8;i++){
      for (std::size_t j=0;j<16; j++){
         add++;
         outfile <<  (int)RAM.get(add) << splitSymbol;
      }
   }*/
   for(int i=0;i<nImp.size();i++){
      outfile <<  (int)RAM.get(nImp[i]+1) << splitSymbol;
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
   /*std::cout << (int)(keystates[SDLK_SPACE] &&
               keystates[SDLK_RIGHT]) << " ";  //Disparo derecha
   std::cout << (int)(keystates[SDLK_SPACE]
               && keystates[SDLK_LEFT]) << " ";  //Disparo izquierda
   std::cout << (int)(keystates[SDLK_SPACE] &&
               !keystates[SDLK_RIGHT] &&
               !keystates[SDLK_LEFT]) << " ";  //Disparo normal

   std::cout << (int)keystates[SDLK_UP] << " ";
   std::cout << (int)keystates[SDLK_LEFT] << " ";
   std::cout << (int)keystates[SDLK_RIGHT] << "\n";*/
}

///////////////////////////////////////////////////////////////////////////////
/// MAIN PROGRAM
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
   // Check input parameter
   if (argc != 2)
      usage(argv[0]);

   // Create alei object.
   alei.setInt("random_seed", 0);
   alei.setFloat("repeat_action_probability", 0);
   alei.setBool("display_screen", true);
   alei.setBool("sound", true);
   alei.loadROM(argv[1]);

   // Init
   //restartNimp(); //Para usar toda la RAM
   srand(time(NULL));
   lastLives = alei.lives();
   totalReward = .0f;
   initVector(currentState);
   initVector(changes);

   // Main loop
   //alei.act(PLAYER_A_FIRE);
   int step;
   for (step = 0; 
        !alei.game_over() && !gameOverManual && step < maxSteps; 
        ++step) 
   {
      totalReward += agentStep();   //Movimiento
      cls();
      showRAM();

      saveStateInFile();
      compareRAM();
   }

   std::cout << "Steps: " << step << std::endl;
   std::cout << "Reward: " << totalReward << std::endl;
   showChanges();

   return 0;
}