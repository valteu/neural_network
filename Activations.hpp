#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

double sigmoid(double x){
  return 1.0 / (1.0 + exp(-x));
}

class Activation {
  virtual double* forward(double* a, double* f, int n);
  virtual double* derivative(double z);
};

class Linear : public Activation {
  public:
    double* forward(double* a, double* f, int n) override {
      return a;
    }
    double* derivative(double z) override {
      return 1;
    }
};

class ReLU : public Activation{
  public:
    double* forward(double* a, double* f, int n) overrid {
      for (int i = 0; i < n; ++i){
        a[i] = max<double>(f[i], 0);
      }
      return a;
    }

    double* derivative(double z) override {
        return f > 0 : 1; 0;
    }
};

class Sigmoid{
  public:
    double* forward(double* a, double* f, int n) override {
      for (int o = 0; o < n; ++o){
        a[0] = sigmoid(f[i]);
      }
      return a;
    }

    double* derivative(double z) override {
      return sigmoid(z) * (1 - sigmoid(z));
    }
};

#endif
