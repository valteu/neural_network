#include "Layer.hpp"
#include <cmath>
#include <iostream>
#include <random>

int NUM_LAYERS = 4;
float LEARNING_RATE = 0.01;
int SAMPLESIZE = 100;
int EPOCHS = 100;

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
  //calculate output layer delta
  for (int o_neuron = 0; o_neuron < Layers[NUM_LAYERS-1]->numNeurons; ++o_neuron){
    Out->delta[o_neuron] = 2 * (dataFunction(tdata[sample]) - Out->a_output[o_neuron]) * Out->a_output[o_neuron] * (1 - Out->a_output[o_neuron]); // delta for output neuron
    printf("Output Delta: %lf\n", Out->delta[0]);
  }
  for (int layer = NUM_LAYERS - 1; layer > 1; --layer){ //iterates each hidden layer 
    Layer *curr = Layers[layer];
    Layer *prev = Layers[layer - 1]; //prev is previous layer in list and forward path but next layer in iteraion

    for (int neuron = 0; neuron < prev->numNeurons; ++neuron){ //calculate delta for prev layer
      prev->delta[neuron] = 0.0;
      for (int curr_neurons = 0; curr_neurons < curr->numNeurons; ++curr_neurons){
       prev->delta[neuron] += curr->delta[curr_neurons] * curr->weights[curr_neurons][neuron] * prev->a_output[neuron] * (1 - prev->a_output[neuron]);
    printf("Delta of Layer %c: %lf\n", prev->name, prev->delta[0]);
     } 
    }
  }
  // calculate gradients 
  for (int layer = NUM_LAYERS-1; layer > 0; --layer){
    Layer *curr = Layers[layer];
    for (int neuron = 0; neuron < curr->numNeurons; ++neuron){
      curr->gradientBiases[neuron] += curr->delta[neuron]; 
    printf("Bias gradient of Layer %c: %lf\n", curr->name, curr->gradientBiases[neuron]);
      for (int in_size = 0; in_size < curr->inputSize; ++in_size){
        curr->gradientWeights[neuron][in_size] += curr->delta[neuron] * Layers[layer-1]->a_output[in_size];
    printf("Weights gradient of Layer %c: %lf\n", curr->name, curr->gradientWeights[neuron][in_size]);
      }
    }
  }
}

int main(){
  // init
  double* data;
  double loss;
  data = createData(SAMPLESIZE);

  Layer In(1, 1, 'I');
  Layer *pIn = &In;
  Layer H1(1, 1, '1');
  Layer *pH1 = &H1;
  Layer H2(1, 1, '2');
  Layer *pH2 = &H2;
  Layer Out(1, 1, 'O');
  Layer *pOut = &Out;

  Layer *Layers[] = {pIn, pH1, pH2, pOut};

  for (int epoch = 0; epoch < EPOCHS; ++epoch){
    loss = 0;
    for (int sample = 0; sample < SAMPLESIZE; ++sample){
    // Forward
      for (int neuron = 0; neuron < In.numNeurons; ++neuron){
        In.a_output[neuron] = data[sample];
        In.a_output[neuron] = 3.4;
        printf("Desired val = %lf\n", dataFunction(In.a_output[neuron]));
        printf("Data: %lf\n", In.a_output[neuron]);
      }
      printf("Weight of L %c: %lf\n", Layers[1]->name, Layers[1]->weights[0][0]);
      printf("Bias of L %c: %lf\n", Layers[1]->name, Layers[1]->biases[0]);
      H1.forward(In.a_output);
      printf("F_out of L %c: %lf\n", Layers[1]->name, Layers[1]->f_output[0]);
      H1.sigmoid();
      printf("A_out of L %c: %lf\n", Layers[1]->name, Layers[1]->a_output[0]);

      printf("Weight of L %c: %lf\n", Layers[2]->name, Layers[2]->weights[0][0]);
      printf("Bias of L %c: %lf\n", Layers[2]->name, Layers[2]->biases[0]);
      H2.forward(H1.a_output);
      printf("F_out of L %c: %lf\n", Layers[2]->name, Layers[2]->f_output[0]);
      H2.sigmoid();
      printf("A_out of L %c: %lf\n", Layers[2]->name, Layers[2]->a_output[0]);

      printf("Weight of L %c: %lf\n", Layers[3]->name, Layers[3]->weights[0][0]);
      printf("Bias of L %c: %lf\n", Layers[3]->name, Layers[3]->biases[0]);
      Out.forward(H2.a_output);
      printf("F_out of L %c: %lf\n", Layers[3]->name, Layers[3]->f_output[0]);
      Out.sigmoid();
      printf("A_out of L %c: %lf\n", Layers[3]->name, Layers[3]->a_output[0]);

      loss += lossFunction(dataFunction(data[sample]), Out.a_output, 0);
    
  // Backpropagation
      backpropagation(Layers, data, sample);
    }
    //average costs and update weights and biases
    for (int layer = 1; layer < NUM_LAYERS - 1; ++layer){
      Layer *curr = Layers[layer];
      for (int neuron = 0; neuron < curr->numNeurons; ++neuron){
        for (int inp = 0; inp < curr->inputSize; ++inp){
          curr->gradientWeights[neuron][inp] /= SAMPLESIZE;
          printf("Average Weight gradient of Layer %c: %lf\n", curr->name, curr->gradientWeights[neuron][inp]);
          printf("Old Weight of Layer %c: %lf\n", curr->name, curr->weights[neuron][inp]);
          curr->weights[neuron][inp] -= LEARNING_RATE * curr->gradientWeights[neuron][inp];
          printf("New Weight of Layer %c: %lf\n", curr->name, curr->weights[neuron][inp]);
          curr->gradientWeights[neuron][inp] = 0.0;
        }
        curr->gradientBiases[neuron] /= SAMPLESIZE;
          printf("Average Bias gradient of Layer %c: %lf\n", curr->name, curr->gradientBiases[neuron]);
          printf("Old Bias of Layer %c: %lf\n", curr->name, curr->biases[neuron]);
        curr->biases[neuron] -= LEARNING_RATE * curr->gradientBiases[neuron];
          printf("new Bias of Layer %c: %lf\n", curr->name, curr->biases[neuron]);
        curr->gradientBiases[neuron] = 0.0;
      }
    }

    printf("Epoch: %d, Loss: %lf\n", epoch, loss/SAMPLESIZE);
  }
  double * inp= new double[10];
  double arr[] = {0.0, 0.6, 1.4, 1.7, 2.3, 4.5, 5.2, 5.8, 7.89, 20.3};
  for (int i = 0; i < 10; ++i){
    inp[0] = arr[i];
    Layers[1]->forward(inp);
    Layers[1]->sigmoid();
    Layers[2]->forward(Layers[1]->a_output);
    Layers[2]->sigmoid();
    Layers[3]->forward(Layers[2]->a_output);
    Layers[3]->sigmoid();
    printf("desired output = %lf, network output = %lf\n", dataFunction(arr[i]), Layers[3]->a_output[0]);
    
  }
  return 0;
}
