#include "Layer.hpp"
#include <cmath>
#include <iostream>
#include <random>

int NUM_LAYERS = 4;
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
  // Forward
  for (int i = 0; i < BATCHES; ++i){
    H1.forward(tdata);
    H1.sigmoid(H1.f_output);
    H2.forward(H1.a_output);
    H2.sigmoid(H2.f_output);
    Out.forward(H2.a_output);
    Out.sigmoid(Out.f_output);
    loss = lossFunction(dataFunction(tdata[i]), Out.a_output, 0);
    printf("Loss: %lf", loss);

    // Backpropagation
    for (int o_neuron = 0; o_neuron < Out.numNeurons; ++o_neuron){
      Out.delta[o_neuron] = 2 * (tdata[i] - Out.a_output[o_neuron]); // delta for output neuron
      std::cout << "delta: "<< Out.delta[o_neuron] << std::endl;
    }
    for (int layer = NUM_LAYERS - 1; layer > 2; --layer){ //iterates each layer
      Layer curr = Layers[layer];
      Layer prev = Layers[layer - 1]; //prev is previous layer in list and forward path but next layer in iteraion

      std::cout << "3.5" << std::endl;
      for (int neuron = 0; neuron < curr.numNeurons; ++neuron){ // update weights
      std::cout << "3.6" << std::endl;
        for (int inp = 0; inp < curr.inputSize; ++inp){
        std::cout << curr.delta[neuron] << " "<< neuron<< std::endl;
        std::cout << prev.a_output[neuron] << " "<< neuron << " " << inp << std::endl;
        std::cout << curr.weights[0][0] << " "<< neuron<< std::endl;

          curr.weights[neuron][inp] -= curr.delta[neuron] * LEARNING_RATE * prev.a_output[neuron]; 
          std::cout << "3.8" << std::endl;
        }
        curr.biases[neuron] -= curr.delta[neuron] * LEARNING_RATE; // update bias
      }
      std::cout << "4" << std::endl;
      for (int neuron = 0; neuron < prev.numNeurons; ++neuron){ //calculate delta for prev layer
       for (int curr_neurons = 0; curr_neurons < curr.numNeurons; ++curr_neurons){
         prev.delta[neuron] += curr.delta[curr_neurons] * curr.weights[curr_neurons][neuron] * (1 - curr.a_output[curr_neurons]);
         std::cout << "delta: "<< prev.delta[neuron] << std::endl;
       } 
      }
    }
  }
  delete[] tdata;
  //delete[] loss;
  return 0;
}
