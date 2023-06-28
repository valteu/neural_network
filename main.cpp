#include "Layer.hpp"
#include <cmath>
#include <iostream>
#include <random>

int NUM_LAYERS = 4;
float LEARNING_RATE = 0.1;
int BATCHES = 1000;
int BATCHSIZE = 1000;

double dataFunction(double x){ // intervall [0, 6] interesting
  return sin(x) + 0.5 * cos(2 * x) + 0.3 * sin(3*x) + 0.2 * cos(4* x) + 0.1 * sin(5*x) + 0.1 * cos(6*x);
}

double* createData(int samples){ //create data array of trainigsdata
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> distribution(1.0, 6.0);

  double* data = new double[samples];
  for (int i = 0; i < samples; ++i){
    double x = distribution(gen);
    data[i] = x;
  }
  return data;
}

double lossFunction(double data, double* nn_out, int data_idx){
  double loss = (dataFunction(data) - nn_out[0]) * (dataFunction(data) - nn_out[0]);
  return loss;
}

int main(){
  // init
  double* tdata;
  double loss;

  Layer In(1, BATCHES, 'I');
  Layer *pIn = &In;
  Layer H1(1, 4, '1');
  Layer *pH1 = &H1;
  Layer H2(4, 4, '2');
  Layer *pH2 = &H2;
  Layer Out(4, 1, 'O');
  Layer *pOut = &Out;

  Layer *Layers[] = {pIn, pH1, pH2, pOut};

  for (int batch = 0; batch < BATCHES; ++batch){
  In.a_output = tdata;
  // Forward
  for (int sample = 0; sample < BATCHSIZE; ++sample){
      H1.forward(tdata);
      H1.sigmoid(H1.f_output);

      H2.forward(H1.a_output);
      H2.sigmoid(H2.f_output);

      Out.forward(H2.a_output);
      Out.sigmoid(Out.f_output);

    loss = lossFunction(dataFunction(tdata[sample]), Out.a_output, 0);
    printf("Loss: %lf\n", loss);

    // Backpropagation
    for (int o_neuron = 0; o_neuron < pOut->numNeurons; ++o_neuron){
      Out.delta[o_neuron] = 2 * (dataFunction(tdata[sample]) - pOut->a_output[o_neuron]) * pOut->a_output[o_neuron] * (1 - pOut->a_output[o_neuron]); // delta for output neuron
      }
    for (int layer = NUM_LAYERS - 1; layer > 0; --layer){ //iterates each layer
      Layer *curr = Layers[layer];
      Layer *prev = Layers[layer - 1]; //prev is previous layer in list and forward path but next layer in iteraion

      for (int neuron = 0; neuron < curr->numNeurons; ++neuron){ // update weights
        for (int inp = 0; inp < curr->inputSize; ++inp){
          curr->gradientWeights[neuron][inp] += curr->delta[neuron]* prev->a_output[neuron]; 
        }
        curr->gradientBiases[neuron] += curr->delta[neuron]; // update bias
      }
      for (int neuron = 0; neuron < prev->numNeurons; ++neuron){ //calculate delta for prev layer
        prev->delta[neuron] = 0.0;
        for (int curr_neurons = 0; curr_neurons < curr->numNeurons; ++curr_neurons){
         prev->delta[neuron] += curr->delta[curr_neurons] * curr->weights[curr_neurons][neuron] * prev->a_output[neuron] * (1 - prev->a_output[neuron]);
       } 
      }
    }
  }
  for (int layer = NUM_LAYERS; layer > 0; --layer){
    for (int neuron = 0; neuron < Layers[layer]->numNeurons; ++neuron){
      Layers[layer]->biases[neuron] -= LEARNING_RATE * Layers[layer]->gradientBiases[neuron] / BATCHSIZE;
      for (int in_size = 0; in_size < Layers[layer]->inputSize; ++in_size){
        Layers[layer]->weights[neuron][in_size] -= LEARNING_RATE * Layers[layer]->gradientWeights[neuron][in_size] / BATCHSIZE;
      }
    }
  }
}
  return 0;
}
