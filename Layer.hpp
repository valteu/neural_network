#ifndef LAYER_HPP
#define LAYER_HPP

#include <cmath>


class Layer{
  public:
    double** weights;
    double* biases;
    double* activation;
    double* pre_activation;
    double* delta;

    double** gradientWeights;
    double* gradientBiases;

    int outputSize;
    int inputSize;

    Layer(int, int);
    void forward(double* input);
    void backward();
    virtual void activate() = 0;

    ~Layer();
};

class Linear : public Layer {
  public: 
    Linear(int, int);
    void activate();
    double derivative(double);
};
class ReLU : public Layer {
  public: 
    ReLU(int, int);
    void activate();
    double derivative(double);
};
class Sigmoid : public Layer {
  public: 
    Sigmoid(int, int);
    void activate();
    double derivative(double);
    double sigmoid(double x){
      return 1.0 / (1.0 + exp(-x));
    }
};

#endif

