#include "Layer.hpp"
#include <cmath>
#include <iostream>
#include <random>

int NUM_LAYERS = 4;
float LEARNING_RATE = 0.1;
int SAMPLESIZE = 10;

void printArray(double arr[], int len){
  for(int i = 0; i < len; ++i){
    printf("lop: %lf\n", arr[i]);
  }
}
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

void backpropagation(Layer *Layers[], double* tdata, int sample){
  Layer *Out = Layers[NUM_LAYERS - 1];
  //calculate output laeyr delta
  for (int o_neuron = 0; o_neuron < Layers[NUM_LAYERS-1]->numNeurons; ++o_neuron){
    Out->delta[o_neuron] = 2 * (dataFunction(tdata[sample]) - Out->a_output[o_neuron]) * Out->a_output[o_neuron] * (1 - Out->a_output[o_neuron]); // delta for output neuron
    //std::cout << Out->delta[o_neuron] << "\n";
  }
  for (int layer = NUM_LAYERS - 1; layer > 0; --layer){ //iterates each hidden layer 
    Layer *curr = Layers[layer];
    Layer *prev = Layers[layer - 1]; //prev is previous layer in list and forward path but next layer in iteraion

    for (int neuron = 0; neuron < prev->numNeurons; ++neuron){ //calculate delta for prev layer
      prev->delta[neuron] = 0.0;
      for (int curr_neurons = 0; curr_neurons < curr->numNeurons; ++curr_neurons){
        std:: cout << "cdelta " << curr->delta[curr_neurons] << " cweights " << curr->weights[curr_neurons][neuron] << " pAOut " << prev->a_output[neuron] << "\n"; 
       prev->delta[neuron] += curr->delta[curr_neurons] * curr->weights[curr_neurons][neuron] * prev->a_output[neuron] * (1 - prev->a_output[neuron]);
    std::cout << "delta " << prev->delta[neuron] << "\n";
     } 
    }
  }
  // calculate gradients 
  for (int layer = NUM_LAYERS-1; layer > 0; --layer){
    Layer *curr = Layers[layer];
    for (int neuron = 0; neuron < curr->numNeurons; ++neuron){
      curr->gradientBiases[neuron] += curr->delta[neuron]; 
      //printf("Gradient_update: %lf; delta: %lf\n", curr->gradientBiases[neuron], curr->delta[neuron]);
      for (int in_size = 0; in_size < curr->inputSize; ++in_size){
        curr->gradientWeights[neuron][in_size] += curr->delta[neuron] * Layers[layer-1]->a_output[in_size];
      }
    }
  }
}

int main(){
  // init
  double* data;
  double loss;
  data = createData(SAMPLESIZE);

  Layer In(1, 4, 'I');
  Layer *pIn = &In;
  Layer H1(1, 4, '1');
  Layer *pH1 = &H1;
  Layer H2(4, 4, '2');
  Layer *pH2 = &H2;
  Layer Out(4, 1, 'O');
  Layer *pOut = &Out;

  Layer *Layers[] = {pIn, pH1, pH2, pOut};

  for (int epoch = 0; epoch < 1; ++epoch){
    loss = 0;
    for (int sample = 0; sample < SAMPLESIZE; ++sample){
    // Forward
      for (int neuron = 0; neuron < In.numNeurons; ++neuron){
        In.a_output[neuron] = data[sample];
      }
      H1.forward(In.a_output);
      H1.sigmoid(H1.f_output);
      //printf("H1\n");
      //printArray(H1.a_output, H1.numNeurons);

      H2.forward(H1.a_output);
      H2.sigmoid(H2.f_output);

      //printf("H2\n");
      //printArray(H2.a_output, H2.numNeurons);
      Out.forward(H2.a_output);
      Out.sigmoid(Out.f_output);

      //printf("Out\n");
      //printArray(Out.a_output, Out.numNeurons);
      loss += lossFunction(dataFunction(data[sample]), Out.a_output, 0);
    
  // Backpropagation
      backpropagation(Layers, data, sample);
    }
    //average costs and update weights and biases
    for (int layer = 0; layer < NUM_LAYERS - 1; ++layer){
      Layer *curr = Layers[layer];
      for (int neuron = 0; neuron < curr->numNeurons; ++neuron){
        for (int inp = 0; inp < curr->inputSize; ++inp){
          curr->gradientWeights[neuron][inp] /= SAMPLESIZE;
          curr->weights[neuron][inp] -= LEARNING_RATE * curr->gradientWeights[neuron][inp];
        }
        curr->gradientBiases[neuron] /= SAMPLESIZE;
        printf("Bias nudge: %lf\n", curr->gradientBiases[neuron]);
        curr->biases[neuron] -= LEARNING_RATE * curr->gradientBiases[neuron];
      }
    }

    printf("Epoch: %d, Loss: %lf\n", epoch, loss/SAMPLESIZE);
  }
  return 0;
}
