#include "Layer.hpp"
#include <iostream>
#include <cmath>


Layer::Layer(int n_neurons, int inSize){
    numNeurons = n_neurons;
    inputSize = inSize;
    f_output = new double[numNeurons];
    a_output = new double[numNeurons];

    weights = new double*[numNeurons];
      for (int i = 0; i < numNeurons; ++i){
        weights[i] = new double[inputSize];

        for (int ii = 0; ii < inputSize; ++ii){
          weights[i][ii] = (double)rand() / RAND_MAX - 0.5;
      }
    }
  biases = new double[numNeurons];
    for (int i = 0; i < inputSize; ++i){
      biases[i] = (double)rand() / RAND_MAX - 0.5;
    }
  }

  void Layer::forward(double* inputData){
    double temp = 0;
    for (int n = 0; n < numNeurons; ++n){ //calculate cross product and add bias
      for (int i = 0; i < inputSize; ++i){
        temp += weights[n][i] * inputData[i];
      }
      f_output[n] = temp + biases[n];
    }
  }
  
  void Layer::relu(double* data){
    for (int i = 0; i < numNeurons; ++i){
      a_output[i] = max<double>(data[i], 0);
    }
   }

  Layer::~Layer(){
    for (int i = 0; i < numNeurons; ++i){
      delete[] weights[i];
    }
    delete[] weights;
    delete[] biases;
    delete[] f_output;
    delete[] a_output;
  }
