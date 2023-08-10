#include <cmath>
#include "Activations.hpp"

double* Linear::forward(double* a, double* f, int n) {
    for (int i = 0; i < n; ++i) {
        a[i] = f[i];
    }
    return a;
}

double Linear::derivative(double z) {
    return 1.0;
}

double* ReLU::forward(double* a, double* f, int n) {
    for (int i = 0; i < n; ++i) {
        a[i] = std::max<double>(f[i], 0);
    }
    return a;
}

double ReLU::derivative(double z) {
    return z > 0 ? 1 : 0;
}

double* Sigmoid::forward(double* a, double* f, int n) {
    for (int o = 0; o < n; ++o) {
        a[o] = sigmoid(f[o]);
    }
    return a;
}

double Sigmoid::derivative(double z) {
    return sigmoid(z) * (1 - sigmoid(z));
}

