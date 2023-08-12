#include <cmath>
#include <iostream>
#include <random>

#include "Network.hpp"
#include "Layer.hpp"

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

  //
/*
  Layer ** layout = new Layer*[4];

  layout[0] = new ReLU(1, 128);
  layout[1] = new ReLU(128, 128);
  layout[2] = new ReLU(128, 128);
  layout[3] = new ReLU(128, 1);
  */
  Layer* layout[4] = {
    new ReLU(1, 12), // input layer
   // new ReLU(128, 128),  // first hidden layer
    new ReLU(12, 12),  // second hidden layer
    new Sigmoid(12, 1),  // output layer
  };
  //create network with given layout
  Network network = Network(layout, 3);
  //train network
  network.train(EPOCHS, SAMPLES, data, targets, LEARNING_RATE); 
  //test network
  double tests[] = {0, 0.1, 0.4, 0.6, 1.0, 0.1};
  network.test(tests, tests, 6);
  //delete[] targets;
  return 0;
}
