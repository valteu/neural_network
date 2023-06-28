#ifndef LAYER_H
#define LAYER_H

template <typename T>
T max(T a, T b){
  if (a > b) return a;
  return b;
}

class Layer{
  public:
    double** weights;
    double* biases;
    double* f_output;
    double* a_output;
    double* delta;

    double** gradientWeights;
    double* gradientBiases;
    char name;

    int numNeurons;
    int inputSize;

    Layer(int, int, char);
    void forward(double* input);
    void relu(double* data);
    void sigmoid(double* data);

    ~Layer();
};

#endif

