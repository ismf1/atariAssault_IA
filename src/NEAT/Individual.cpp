#include <NEAT/Individual.hpp>
#include <NeuralNetworkv2/NeuralNet.hpp>
#include <NeuralNetworkv2/Functions.hpp>
#include <ale_interface.hpp>

const std::vector<int16_t> TOPOLOGY { 128, 128, 64, 5 };
ActFunc  actfRelu  { Functions::relu, Functions::reluD };
ActFunc  actfSigm  { Functions::sigm, Functions::sigmD };
VecActFunc  actf  {
    actfRelu,
    actfRelu,
    actfSigm
};

Individual::Individual(const VecWeights &weights) {
    nn = NNet(TOPOLOGY, actf, weights);
}

Individual::Individual() {
    nn = NNet(TOPOLOGY, actf);
}

Individual::Individual(const Individual &o) {
    nn = o.nn;
    fit = o.fit;
    reward = o.reward;
    movesLeft = o.movesLeft;
    movesRight = o.movesRight;
}

const std::vector<double> Individual::getState(ALEInterface &alei) {
    const auto& RAM = alei.getRAM();
    std::vector<double> result(RAM.size());

    for (size_t i = 0; i < RAM.size(); i++) {
        double value = (double)RAM.get(i + 1);
        result[i] = value / 255;
    }

    return result;
}

double Individual::agentStep(ALEInterface &alei) {
   
   double reward = 0;
   std::vector<double> res = nn.predict(getState(alei));

   if(res[0] > 0.5) reward += alei.act(PLAYER_A_RIGHTFIRE);
   if(res[1] > 0.5) reward += alei.act(PLAYER_A_LEFTFIRE);
   if(res[2] > 0.5) reward += alei.act(PLAYER_A_UPFIRE);
   if(res[3] > 0.5) {
        movesLeft++;
        reward += alei.act(PLAYER_A_LEFT);
   } 
   if(res[4] > 0.5) {
        movesRight++;
        reward += alei.act(PLAYER_A_RIGHT);
   }
   
   
   return (reward + alei.act(PLAYER_A_NOOP));
}

bool Individual::operator<(const Individual& other) const {
    return fit > other.fit;
}

Individual Individual::crossover(const Individual &other, double mutateRate) {

    VecWeights weights(nn.size());

    for (size_t i = 0; i < nn.size(); i++) {
        auto lw1 = nn[i].getWeights();
        auto lw2 = other.nn[i].getWeights();
        std::vector<std::vector<double>> lw(lw1.size());

        for (size_t j = 0; j < lw1.size(); j++) {

            std::vector<double> nw(lw1[j].size());

            for (size_t k = 0; k < lw1[j].size(); k++) {
                auto x = ((double) rand() / (RAND_MAX));                
                nw[k] =  x < mutateRate?
                        Functions::rand(-1, 1) :
                        Functions::randomChoice(lw1[i][j], lw2[i][j]);  
            }

            lw[j] = nw;
        }

        weights[i] = lw;
    }

    return Individual(weights);
}

double Individual::fitness(bool display)
{
    for (size_t i = 0; i < 3; i++) {
        std::cerr.setstate(std::ios_base::failbit);

        ALEInterface alei;

        alei.setFloat("repeat_action_probability", 0);
        alei.setBool("display_screen", display);
        alei.setBool("sound", false);
        alei.loadROM("assets/supported/assault.bin");

        std::cerr.clear();

        reward = 0; 
        fit = 0;

        double totalReward = 0;
        double step = 0;
        movesLeft = 0;
        movesRight = 0;
        for (step = 0;!alei.game_over();++step)
        {
            totalReward += agentStep(alei); //Movimiento
        }
        reward += totalReward;
        fit += ((movesLeft + movesRight) * 0.1) + totalReward * 1.3;
        //fit = ;
    }

    reward /= 3;
    fit /= 3;
    //fit = step;
    // totalReward = totalReward < 100? -10000 : totalReward;
    // fit = step;
    return reward;
}
