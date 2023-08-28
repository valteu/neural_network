#include <cmath>
#include <iostream>
#include <random>

#include "Network.hpp"
#include "Layer.hpp"

float LEARNING_RATE = 0.1;
int SAMPLES = 1000;
int EPOCHS = 10000;
int LAYERS = 4;

double dataFunction(double x){ 
  return x;
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

  //
  Layer* layout[LAYERS] = {
    new ReLU(1, 12), // input layer
    new ReLU(12, 128),  // first hidden layer
    new ReLU(128, 12),  // second hidden layer
    new Sigmoid(12, 1),  // output layer
  };
  //create network with given layout
  Network network = Network(layout, LAYERS);
  //train network
  network.train(EPOCHS, SAMPLES, data, targets, LEARNING_RATE); 
  //test network
  double tests[] = {0, 0.1, 0.4, 0.6, 1.0, 0.1};
  network.test(tests, tests, 6);

  delete[] targets;
  delete[] data;
  return 0;
}
