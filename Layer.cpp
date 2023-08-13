#include "Layer.hpp"

#include <iostream>
#include <cmath>


Layer::Layer(int inp_size, int out_size)
  : outputSize(out_size), inputSize(inp_size){

  pre_activation = new double[outputSize];
  activation = new double[outputSize];
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
    pre_activation[o] = temp + biases[o];
  }
}


void Layer::backward(bool first, double* targets, Layer* next, Layer* prev){
  // compute output Layer delta
  if (first){
    for (int o = 0; o < outputSize; ++o){
      delta[o] = 2 * (targets[o] - activation[o]) * derivative(activation[o]);
    }
  }
  //compute hidden Layer delta
  else{
    for (int o = 0; o < outputSize; ++o){
      delta[o] = 0.0;
      for (int i = 0; i < next->outputSize; ++i){
        delta[o] += next->delta[i] * next->weights[o][i] * derivative(activation[o]); 
      }
    }
  }
  // update gradients
  for (int o = 0; o < outputSize; ++o){
    gradientBiases[o] = delta[o];
    for (int i = 0; i < inputSize; ++i){
      gradientWeights[i][o] = delta[o] * prev->activation[i];
    }
  }
}


Layer::~Layer(){
  for (int i = 0; i < inputSize; ++i){
    delete[] weights[i];
    delete[] gradientWeights[i];
  }
  delete[] weights;
  delete[] gradientWeights;
  delete[] biases;
  delete[] pre_activation;
  delete[] activation;
  delete[] delta;
  delete[] gradientBiases;
}


Linear::Linear(int inp_size, int out_size) : Layer(inp_size, out_size){}
void Linear::activate() {
    for (int i = 0; i < outputSize; ++i) {
        activation[i] = pre_activation[i];
    }
}
double Linear::derivative(double z) {
    return 1.0;
}


ReLU::ReLU(int inp_size, int out_size) : Layer(inp_size, out_size){}
void ReLU::activate() {
    for (int i = 0; i < outputSize; ++i) {
        activation[i] = std::max<double>(pre_activation[i], 0);
    }
}
double ReLU::derivative(double z) {
    return z > 0 ? 1 : 0;
}

Sigmoid::Sigmoid(int inp_size, int out_size) : Layer(inp_size, out_size){}
void Sigmoid::activate() {
    for (int o = 0; o < outputSize; ++o) {
        activation[o] = sigmoid(pre_activation[o]);
    }
}
double Sigmoid::derivative(double z) {
    return sigmoid(z) * (1 - sigmoid(z));
}
