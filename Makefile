CXX      := -g++
CXXFLAGS := -pedantic-errors -Wall -Wextra -std=c++2a -fconcepts -O3 -funroll-loops -Wno-reorder
LDFLAGS  := -L/usr/lib -L./lib/ale -lale -lSDL -lstdc++ -lm
BUILD    := ./build
OBJ_DIR  := $(BUILD)/obj
APP_DIR  := .
TARGET   := $(basename $(notdir $(main)))
SRC_FILES_NN := $(wildcard ./src/**/*.cpp)
INCLUDE  := -I./include -I./lib/ale/src
SRC      := $(filter-out $(filter-out $(main), $(wildcard ./src/Main/*.cpp)), $(SRC_FILES_NN))
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

trainNN: 
	make main=./src/Main/TrainNN.cpp

trainNN2: 
	make main=./src/Main/TrainNN2.cpp

trainPerceptron: 
	make main=./src/Main/TrainPerceptron.cpp

trainGA: 
	make main=./src/Main/TrainGA.cpp

agentGA:
	make main=./src/Main/AgentGA.cpp

agentManual:
	make main=./src/Main/AgentManual.cpp

agentNN:
	make main=./src/Main/AgentNN.cpp

agentPerceptron:
	make main=./src/Main/AgentPerceptron.cpp
    
clean:
	-@rm -rvf $(OBJ_DIR)/*
	-@rm -rvf TrainNN
	-@rm -rvf TrainNN2
	-@rm -rvf TrainPerceptron
	-@rm -rvf TrainGA
	-@rm -rvf Agent*

info:
	@echo "[*] Object dir:      ${OBJ_DIR}     "
	@echo "[*] Sources:         ${SRC}         "
	@echo "[*] Objects:         ${OBJECTS}     "
	@echo "[*] Dependencies:    ${DEPENDENCIES}"