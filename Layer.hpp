#ifndef LAYER_HPP
#define LAYER_HPP

#include "Activations.hpp"


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

    Activation* activaton;

    Layer(int, int, Activation*);
    void forward(double* input);
    void backward();

    ~Layer();
};

#endif

