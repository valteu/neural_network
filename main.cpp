#include "Layer.hpp"
#include <cmath>
#include <iostream>
#include <random>

double dataFunction(double x){ // intervall [0, 6] interesting
  return sin(x) + 0.5 * cos(2 * x) + 0.3 * sin(3*x) + 0.2 * cos(4* x) + 0.1 * sin(5*x) + 0.1 * cos(6*x);
}

double* createData(int samples){ //create data array of trainigsdata
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> distribution(1.0, 0.6);

  double* data = new double[samples * 2];
  for (int i = 0; i+1 < samples; i += 2){
    double x = distribution(gen);
    data[i] = x;
    data[i+1] = dataFunction(x);
  }
  return data;
}

int main(){

  double* tdata = createData(50000);

  printf("1\n");
  for (int i = 0; i < 50000; ++i){
    printf("%lf \n", tdata[i]);
  }
  Layer H1(4, 2);
  Layer H2(4, 4);
  Layer Out(2, 4);

  H1.forward(tdata);
  //printf("%lf \n", H1.f_output[0]);
  H1.relu(H1.f_output);
  //printf("%lf \n", H1.a_output[0]);

  H2.forward(H1.a_output);
  H2.relu(H2.f_output);

  Out.forward(H2.a_output);
  Out.relu(Out.f_output);
  printf("%lf \n", Out.a_output[0]);
  printf("%lf \n", Out.a_output[1]);
  delete[] tdata;
}
