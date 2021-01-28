## 1. Dependencias

### 1. 1 Librerías
```bash
sudo apt-get install libsdl1.2-dev libsdl-gfx1.2-dev libsdl-image1.2-dev cmake
```
### 1.2 Instalar ALE
```bash
cd lib/ale/ && mkdir build && cd build
cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON ..
make -j 8
```
## 2. Compilación 
- **Entrenamiento Red Neuronal Iván:**
    ```bash
    make trainNN 
    ```
- **Entrenamiento Red Neuronal Carlos:**
    ```bash
    make trainNN2 
    ```
- **Entrenamiento Perceptrón:**
    ```bash
    make trainPerceptron
    ```
- **Entrenamiento Algoritmo Genético:**
    ```bash
    make trainGA 
    ```
- **Agente Red Neuronal:**
    ```bash
    make agentNN
    ```
- **Agente Perceptron:**
    ```bash
    make agentPerceptron
    ```
- **Agente Manual:**
    ```bash
    make agentManual
    ```
- **Agente Algoritmo Genético:**
    ```bash
    make agentGA
    ```
## Ejecución  
- **Entrenamiento Red Neuronal Iván:**
```bash
    LD_LIBRARY_PATH="./lib/ale" ./TrainNN
```
- **Entrenamiento Red Neuronal Carlos:**
```bash 
    LD_LIBRARY_PATH="./lib/ale" ./TrainNN2 <dataSet> <archivoGuardarNN> <iteraciones> <learningRate>
```
- **Entrenamiento Perceptrón:**
```bash 
    LD_LIBRARY_PATH="./lib/ale" ./TrainPerceptron <dataSet>
```
- **Entrenamiento Algoritmo Genético:**
```bash 
    LD_LIBRARY_PATH="./lib/ale" ./TrainGA 
```
- **Agente Red Neuronal:**
```bash
    LD_LIBRARY_PATH="./lib/ale" ./AgentNN <binarioJuego> <archivoModeloNN> <tipo> <archivoEscalador>
```
- **Agente Perceptron:**
```bash
    LD_LIBRARY_PATH="./lib/ale" ./AgentPerceptron <binarioJuego> <nPerceptron>
```
- **Agente Manual:**
```bash
    LD_LIBRARY_PATH="./lib/ale" ./AgentManual <binarioJuego>
```
- **Agente Algoritmo Genético:**
```bash
    LD_LIBRARY_PATH="./lib/ale" ./AgentGA <binarioJuego> <archivoModeloNN>       
```

## Ejemplo

Al entrenar la red ("make trainNN" o "make trainNN2") se guardará un fichero "scaler.txt" y "NeuralNetwork.txt", esto ficheros serán, respectivamente, el escalador y el modelo que pasaremos como argumentos al ejecutar el bot. 
Por otro lado, el <tipo> de la red que nos pedirá al ejecutar (LD_LIBRARY_PATH="./lib/ale" ./AgentNN <binarioJuego> <archivoModeloNN> <tipo> <archivoEscalador>) es "ivan", en caso de haber entrenado la Red Neuronal Iván, o "carlos", en caso de haber entrenado la Red Neuronal Carlos.

## Autores
Carlos Garrido Marín (cgm164@alu.ua.es)
Raquel González Barberá (rgb64@gcloud.ua.es)
Iván San Martín Fernández (ismf1@gcloud.ua.es)