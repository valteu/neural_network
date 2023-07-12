#include "Layer.hpp"
#include <iostream>
#include <cmath>


Layer::Layer(int inp_size, int num_neurons){
  numNeurons = num_neurons;
  inputSize = inp_size;
  f_output = new double[numNeurons];
  a_output = new double[numNeurons];
  delta = new double[numNeurons];

  gradientBiases = new double[numNeurons];
  gradientWeights = new double*[numNeurons];
  for (int i = 0; i < numNeurons; ++i){
    gradientWeights[i] = new double[inputSize];
    gradientBiases[i] = 0.0;
    for (int ii = 0; ii < inputSize; ++ii){
      gradientWeights[i][ii] = 0.0;
    }
  }

  weights = new double*[numNeurons];
  for (int i = 0; i < numNeurons; ++i){
    weights[i] = new double[inputSize];
    for (int ii = 0; ii < inputSize; ++ii){
      weights[i][ii] = (double)rand() / RAND_MAX - 0.5;
    }
  }
  biases = new double[numNeurons];
    for (int i = 0; i < numNeurons; ++i){
      biases[i] = (double)rand() / RAND_MAX - 0.5;
    }
  }

  void Layer::forward(double* inputData){
    for (int n = 0; n < numNeurons; ++n){ //calculate cross product and add bias
      double temp = 0.0;
      for (int i = 0; i < inputSize; ++i){
        temp += weights[n][i] * inputData[i];
      }
      f_output[n] = temp + biases[n];
    }
  }
  
  void Layer::relu(){
    for (int i = 0; i < numNeurons; ++i){
      a_output[i] = max<double>(f_output[i], 0);
    }
  }
  void Layer::sigmoid(){
    for (int i = 0; i < numNeurons; ++i){
      a_output[i] = 1 / (1 + exp(-f_output[i]));
    }
  }
  Layer::~Layer(){
    for (int i = 0; i < numNeurons; ++i){
      delete[] weights[i];
      delete[] gradientWeights[i];
    }
    delete[] weights;
    delete[] gradientWeights;
    delete[] biases;
    delete[] f_output;
    delete[] a_output;
    delete[] delta;

    delete[] gradientBiases;
  }
