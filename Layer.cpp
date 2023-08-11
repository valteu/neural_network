#include "Layer.hpp"
#include "Activations.hpp"

#include <iostream>
#include <cmath>


Layer::Layer(int inp_size, int out_size, Activation* _activaton)
  : outputSize(out_size), inputSize(inp_size), activaton(_activaton){

  f_output = new double[outputSize];
  a_output = new double[outputSize];
  delta = new double[outputSize];

  gradientBiases = new double[outputSize];
  gradientWeights = new double*[inputSize];

  weights = new double*[inputSize];
  biases = new double[outputSize];

  //initialize weights, biases and gradient vectors
  for (int i = 0; i < inputSize; ++i){
    weights[i] = new double[inputSize];
    gradientWeights[i] = new double[outputSize];
    for (int o = 0; o < outputSize; ++o){
      weights[i][o] = (double)rand() / RAND_MAX * 2.0 - 1.0;
      gradientWeights[i][o] = 0.0;
    }
  }

  for (int o = 0; o < outputSize; ++o){
    gradientBiases[o] = 0;
    biases[o] = (double)rand() / RAND_MAX * 2.0 - 1.0;
  }
}

void Layer::forward(double* inputData){
  for (int o = 0; o < outputSize; ++o){ //calculate cross product and add bias
    double temp = 0.0;
    for (int i = 0; i < inputSize; ++i){
      temp += weights[i][o] * inputData[o];
    }
    f_output[o] = temp + biases[o];
  }
  activaton->forward(a_output, f_output, outputSize);
}

/*
void Layer::backward(bool first, double* targets, Layer* prev){
  if (first){
    for (int o = 0; o < outputSize; ++o){
      delta[o] = 2 * (targets[o] - a_output[o]) * activaton.derivative(prev->f_output[o]);
    }
  }

  else{
    for (int o = 0; o < outputSize; ++o){
      delta[o] = 0.0;
      for (int i = 0; i < inputSize; ++i){
        
      }
    }
  }
}
  */

Layer::~Layer(){
  for (int i = 0; i < inputSize; ++i){
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
