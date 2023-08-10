#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include <cmath>
#include <iostream>

template <typename T>
T max(T a, T b){
  return a < b ? b : a;
}

class Activation {
public:
  virtual double* forward(double* a, double* f, int n) = 0;
  virtual double derivative(double z);
  double sigmoid(double x){
    return 1.0 / (1.0 + exp(-x));
  }
};

class Linear : public Activation {
public:
  double* forward(double* a, double* f, int n) override;
  double derivative(double z) override; 
};

class ReLU : public Activation {
public:
  double* forward(double* a, double* f, int n) override;
  double derivative(double z) override; 
};

class Sigmoid : public Activation {
public:
  double* forward(double* a, double* f, int n) override;
  double derivative(double z) override; 
};

#endif

