#include "Layer.hpp"
#include "Network.hpp"
#include <cmath>
#include <iostream>
#include <random>

float LEARNING_RATE = 0.5;
int SAMPLES = 1000;
int EPOCHS = 50000;

double dataFunction(double x){ 
  return x*x;
}

//create and return array and fill it with doubles between 0 and 1
double* createData(int samples){ //create data array of trainigsdata
  double* data = new double[samples];
  for (int i = 0; i < samples; ++i){
    data[i] = (double)rand() / RAND_MAX;
  }
  return data;
}

int main(){
  double* data = createData(SAMPLES);
  //create array of target values
  double* targets = new double[SAMPLES];
  for (int i = 0; i < SAMPLES; ++i){
    targets[i] = dataFunction(data[i]);
  }

  //create network layout: 0. index = size of input, 1. index = number input neurons, 2. index = number second layer neurons, ... 
  int layout[] = {1, 5, 5, 1};
  //create network with given layout
  Network network = Network(layout, sizeof(layout) / sizeof(layout[0]) - 1);
  //train network
  network.train(EPOCHS, SAMPLES, data, targets, LEARNING_RATE); 
  //test network
  double tests[] = {0, 0.1, 0.4, 0.6, 1.0, 0.1};
  network.test(tests, tests, 6);
  //delete[] targets;
  return 0;
}
