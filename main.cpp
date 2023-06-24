#include "Layer.hpp"
#include <cmath>
#include <iostream>
#include <random>

int NUM_LAYERS = 3;
float LEARNING_RATE = 0.1;
int BATCHES = 1000;

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
  /*int size = sizeof(nn_out) / sizeof(nn_out[0]); 
  size = 1;
  printf("%d\n", size);
  double* loss = new double[2];
  printf("g\n");
  for (int i = 0; i < size; ++i){
    loss[i] = (nn_out[i] - data[data_idx + i]) * (nn_out[i] - data[data_idx + i]);
  }
  */
  double loss = (data - nn_out[0]) * (data - nn_out[0]);
  return loss;
}

int main(){
  // init
  double* tdata = createData(BATCHES);
  double loss;

  Layer In(1, BATCHES);
  Layer H1(1, 4);
  Layer H2(4, 4);
  Layer Out(4, 1);

  Layer Layers[] = {In, H1, H2, Out};

  In.a_output = tdata;
  // forward
  for (int i = 0; i < BATCHES; ++i){
    H1.forward(tdata);
    H1.sigmoid(H1.f_output);
    printf("1\n");
    H2.forward(H1.a_output);
    H2.sigmoid(H2.f_output);

    Out.forward(H2.a_output);
    Out.sigmoid(Out.f_output);
    loss = lossFunction(dataFunction(tdata[i]), Out.a_output, 0);
    printf("Loss: %lf", loss);

    printf("2\n");
    // Backpropagation
    double delta = 2 * (tdata[i] - Out.a_output[i]);
    double new_delta = 0;

    for (int layer = NUM_LAYERS; layer > 1; --layer){
    printf("3\n");
      for (int neuron = 0; neuron < Layers[layer].numNeurons; ++neuron){
        for (int weight = 0; weight < Layers[layer].inputSize; ++weight){
          printf("4\n");
         Layers[layer].weights[neuron][weight] *= delta * LEARNING_RATE * Layers[layer-1].a_output[neuron];
        }
      }
      for (int bias = 0; bias < Out.numNeurons; ++ bias){
        Layers[layer].biases[bias] *= delta * LEARNING_RATE;
      }
    printf("4\n");
      for (int neuron = 0; neuron < Layers[layer].numNeurons; ++neuron){
        for (int weight = 0; weight < Layers[layer].inputSize; ++weight){
          new_delta += delta * Layers[layer].weights[neuron][weight] * Layers[layer].a_output[neuron] * (1 - Layers[layer].a_output[neuron]);
        }
      }
      delta = new_delta;
      new_delta = 0;
    }
  printf("NEXT\n");
  }
  delete[] tdata;
  //delete[] loss;
  return 0;
}
