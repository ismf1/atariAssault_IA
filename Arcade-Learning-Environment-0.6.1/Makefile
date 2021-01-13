CXX      := -c++
CXXFLAGS := -pedantic-errors -Wall -Wextra -std=c++2a -fconcepts -O3 -funroll-loops
LDFLAGS  := -L/usr/lib -L. -lale -lSDL -lstdc++ -lm
BUILD    := ./build
OBJ_DIR  := $(BUILD)/obj
APP_DIR  := .
TARGET   := nn
SRC_FILES_NN := $(wildcard ../NeuralNetworkv2/src/*.cpp)
INCLUDE  := -I../NeuralNetworkv2/include/ -I../include -I../neuralNetwork -I../ -I../guardar
SRC      :=                      \
	$(filter-out ../NeuralNetworkv2/src/main.cpp, $(SRC_FILES_NN)) \
	$(filter-out ../neuralNetwork/main.cpp, $(wildcard ../neuralNetwork/*.cpp)) \
	$(wildcard ../guardar/*.cpp) \
	$(wildcard ./minimal_agent_IA_nn_c.cpp)

OBJECTS  := $(SRC:%.cpp=$(OBJ_DIR)/%.o)
DEPENDENCIES := $(OBJECTS:.o=.d)

all: build $(APP_DIR)/$(TARGET)

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -MMD -o $@

$(APP_DIR)/$(TARGET): $(OBJECTS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $(APP_DIR)/$(TARGET) $^ $(LDFLAGS)

-include $(DEPENDENCIES)

.PHONY: all build clean debug release info

build:
	@mkdir -p $(APP_DIR)
	@mkdir -p $(OBJ_DIR)

debug: CXXFLAGS += -DDEBUG -g
debug: all

release: CXXFLAGS += -O2
release: all

clean:
	-@rm -rvf $(OBJ_DIR)/*
	-@rm -rvf $(APP_DIR)/*

info:
	@echo "[*] Object dir:      ${OBJ_DIR}     "
	@echo "[*] Sources:         ${SRC}         "
	@echo "[*] Objects:         ${OBJECTS}     "
	@echo "[*] Dependencies:    ${DEPENDENCIES}"