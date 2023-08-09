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

    int outputSize;
    int inputSize;

    Layer(int, int);
    void forward(double* input);
    void relu();
    void sigmoid();

    ~Layer();
};

#endif

