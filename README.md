------------------------------DEPENDENCIAS-------------------------------
sudo apt-get install libsdl1.2-dev libsdl-gfx1.2-dev libsdl-image1.2-dev cmake
cd lib/ale/ && mkdir build && cd build
cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON ..
make -j 8

--------------------------------COMANDOS---------------------------------
Compilar minimal_agents:
g++ minimal_agent_IA.cpp -o m_IA -L. -lale -lSDL

Ejecutar minimal_agents:
LD_LIBRARY_PATH="." ./m_IA rom/supported/assault.bin

Ejecutar minimal_agent_IA:
LD_LIBRARY_PATH="." ./m_IA rom/supported/assault.bin <n>

Compilar perceptron:
g++ Data.h Data.cpp Perceptron.h Perceptron.cpp main.cpp -o main


-----------------------------FUNCIONAMIENTO--------------------------------
Generación de pesos:
1. Jugar al atari assault para recopilar datos con el comando siguiente:
LD_LIBRARY_PATH="." ./m_IA rom/supported/assault.bin
Esto guardará los datos en el fichero "data.txt".

[OPCIÓN1: perceptrón]
2.1 Entrar en la carpeta perceptron
2.2 Ejecutar ./main que leerá los datos del fichero ../data.txt.
Los pesos generados se guardará en el archivo IA.txt

Lanzar atari assault con pesos del perceptrón
3.1 Mover el archivo IA.txt a la carpeta del atari
3.2 Lanzar el minimal_agent_IA con el siguiente comando:
LD_LIBRARY_PATH="." ./m_IA rom/supported/assault.bin <n>
n = numero del vector de pesos a ejecutar del archivo IA.txt

[OPCIÓN2: multiperceptron network]
En construcción...