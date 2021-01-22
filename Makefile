CXX      := -c++
CXXFLAGS := -pedantic-errors -Wall -Wextra -std=c++2a -fconcepts -O3 -ftree-vectorize -ftree-vectorizer-verbose=1 -msse2 -funroll-loops -fopenmp -mfpmath=sse -march=native
LDFLAGS  := -L/usr/lib -L./lib/ale -lale -lSDL -lstdc++ -lm -ltbb
BUILD    := ./build
OBJ_DIR  := $(BUILD)/obj
APP_DIR  := .
TARGET   := $(basename $(notdir $(main)))
SRC_FILES_NN := $(wildcard ./src/**/*.cpp)
INCLUDE  := -I./include -I./lib/ale/src
SRC      := $(filter-out $(filter-out $(main), $(wildcard ./src/main/*.cpp)), $(SRC_FILES_NN))
OBJECTS  := $(SRC:%.cpp=$(OBJ_DIR)/%.o)
DEPENDENCIES := $(OBJECTS:.o=.d)

all: build $(APP_DIR)/$(TARGET)

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -MMD -o $@

$(APP_DIR)/$(TARGET): $(OBJECTS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $(APP_DIR)/$(TARGET) $^ $(LDFLAGS)

-include $(DEPENDENCIES)

.PHONY: all build clean debug release info train

build:
	@mkdir -p $(APP_DIR)
	@mkdir -p $(OBJ_DIR)

debug: CXXFLAGS += -DDEBUG -g
debug: all

release: CXXFLAGS += -O2
release: all

trainv2: 
	make main=./src/main/trainNeuralNetwork.cpp
train: 
	make main=./src/main/trainNeuralNetworkv2.cpp

run: 
	make main=./src/main/minimal_agent_IA_nn_c.cpp
	LD_LIBRARY_PATH="./lib/ale" ./minimal_agent_IA_nn_c assets/supported/assault.bin $(file) $(type)

clean:
	-@rm -rvf $(OBJ_DIR)/*
	-@rm -rvf trainNeuralNetwork
	-@rm -rvf trainNeuralNetworkv2
	-@rm -rvf nn
	-@rm -rvf trainNeuralNetworkv2.o
	-@rm -rvf minimal_agent_IA_nn_c

info:
	@echo "[*] Object dir:      ${OBJ_DIR}     "
	@echo "[*] Sources:         ${SRC}         "
	@echo "[*] Objects:         ${OBJECTS}     "
	@echo "[*] Dependencies:    ${DEPENDENCIES}"