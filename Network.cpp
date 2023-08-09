#include "Network.hpp"
#include "Layer.hpp"

#include <iostream>


Network::Network(int* layout, int num_layers){
  // layout has size num_layers + 1 to store netInSize and (numNeurons for each layer)
  nlayers = num_layers; 
  netInSize = layout[0];
  netOutSize = layout[nlayers];
  tdata = new double[netInSize];
  tdesired_data = new double[netInSize];
  Layers = new Layer*[nlayers]; 
  for (int layer = 0; layer < nlayers; ++layer){
    Layers[layer] = new Layer(layout[layer], layout[layer + 1]);
  }
}

double Network::squaredLoss(double* desired, double* netOut, int outputSize){
  double loss = 0;
  for (int output = 0; output < outputSize; ++output){
    loss += (desired[output] - netOut[output]) * (desired[output] - netOut[output]);
  }
  return loss;
}

void Network::forwardPass(double* data){
  Layers[0]->forward(data);
  Layers[0]->sigmoid();
  for (int layer = 1; layer < nlayers; ++layer){
    Layer* curr = Layers[layer];
    Layer* prev = Layers[layer - 1];
    curr->forward(prev->a_output);
    curr->sigmoid();
  }
}

void Network::backwardPass(){
  Layer* Out = Layers[nlayers - 1];
  //calculate output layer deltas
  for (int output = 0; output < Out->outputSize; ++output){
    double neuronActive = Out->a_output[output];
    Out->delta[output] = 2 * (tdesired_data[output] - neuronActive) * neuronActive * (1 - neuronActive);
  }
  //iterate each hidden layer
  for (int layer = nlayers - 1; layer > 0; --layer){
    Layer* curr = Layers[layer];
    Layer* prev = Layers[layer - 1];
    //calculate delta for prev layer
    for (int output = 0; output < prev->outputSize; ++output){
      prev->delta[output] = 0.0;
      double prevNeuronActive = prev->a_output[output];
      for (int curr_output = 0; curr_output < curr->outputSize; ++curr_output){
        prev->delta[output] += curr->delta[curr_output] * curr->weights[output][curr_output] * prevNeuronActive * (1 - prevNeuronActive);
      }
    }
  }
  //calculate gradients
  for (int layer = nlayers - 1; layer > 0; --layer){
    Layer* curr = Layers[layer];
    for (int input = 0; input < curr->inputSize; ++input){
      for (int output = 0; output < curr->outputSize; ++output){
        curr->gradientWeights[input][output] += curr->delta[output] * Layers[layer-1]->a_output[input];
        curr->gradientBiases[output] += curr->delta[output];
      }
    }
  }
}

/*
void Network::backwardPass(){
  Layer* Out = Layers[nlayers - 1];
  //calculate output layer deltas
  for (int neuron = 0; neuron < Out->numNeurons; ++neuron){
    double neuronActiv = Out->a_output[neuron];
    Out->delta[neuron] = (tdesired_data[neuron] - neuronActiv);
  }
  //iterate each hidden layer
  for (int layer = nlayers - 1; layer > 0; --layer){
    Layer* curr = Layers[layer];
    Layer* prev = Layers[layer - 1];
    //calculate delta for prev layer
    for (int neuron = 0; neuron < prev->numNeurons; ++neuron){
      prev->delta[neuron] = 0;
      for (int curr_neuron = 0; curr_neuron < curr->numNeurons; ++curr_neuron){
        prev->delta[neuron] = curr->delta[curr_neuron] * curr->weights[curr_neuron][neuron];
      }
    }
  }
  //calculate gradients
  for (int layer = nlayers - 1; layer > 0; --layer){
    Layer* curr = Layers[layer];
    for (int neuron = 0; neuron < curr->numNeurons; ++neuron){
      curr->gradientBiases[neuron] = curr->delta[neuron] * curr->a_output[neuron] * (1.0 - curr->a_output[neuron]);
      for (int inSize = 0; inSize < curr->inputSize; ++inSize){
        curr->gradientWeights[neuron][inSize] = curr->delta[neuron] * Layers[layer-1]->a_output[inSize] * curr->a_output[neuron] * (1.0 - curr->a_output[neuron]);
      }
    }
  }
}
*/
  void Network::updateLayers(int samples, float learning_rate){
  for (int layer = 0; layer < nlayers; ++layer){
    Layer* curr = Layers[layer];
    for (int o = 0; o < curr->outputSize; ++o){
      for (int i = 0; i < curr->inputSize; ++i){
        //update weights
        curr->gradientWeights[i][o] /= samples;
        curr->weights[i][o] += learning_rate * curr->gradientWeights[i][o];
        curr->gradientWeights[i][o] = 0.0;
      }
      //update biases
      curr->gradientBiases[o] /= samples;
      curr->biases[o] += learning_rate * curr->gradientBiases[o];
      curr->gradientBiases[o] = 0.0;
    }
  }
}

void Network::train(int epochs, int samples, double* data, double* desired_data, float learning_rate){
  for (int epoch = 1; epoch <= epochs; ++epoch){
    loss = 0;
    for (int sample = 0; sample < samples; ++sample){
      //resize data to network input size
      for (int netIn = 0; netIn < netInSize; ++netIn){
        tdata[netIn] = data[sample * netInSize + netIn];
      }
      //resize tdesired_data to network output size
      for (int netOut = 0; netOut < netOutSize; netOut ++){
        tdesired_data[netOut] = desired_data[sample * netOutSize + netOut];
      }

      forwardPass(tdata);
      loss += squaredLoss(tdesired_data, Layers[nlayers - 1]->a_output, Layers[nlayers - 1]->outputSize);
    }
    if (epoch % 10000 == 0){
      printf("Epoch: %d, loss: %lf\n", epoch, loss / samples);
    }
    backwardPass();
    updateLayers(samples, learning_rate);
  }
}

void Network::test(double* inputs, double* targets, int num){
  double* tests = new double[netInSize];
  for (int sample = 0; sample < num; ++sample){
    for (int netIn = 0; netIn < netInSize; ++netIn){
      tests[netIn] = inputs[sample * netInSize + netIn];
    }
    forwardPass(tests);
    printf("Test x = %lf, Network output = %lf, target = %lf\n", tests[0], Layers[nlayers - 1]->a_output[0], targets[sample]);
  }
}

Network::~Network(){
  for (int layer = 0; layer < nlayers; ++layer){
    delete Layers[layer];
  }
  delete[] Layers;
  delete[] tdata;
  delete[] tdesired_data;
}
  
