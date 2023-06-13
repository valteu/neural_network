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
  for (int i = 0; i < samples; ++i){
    double x = distribution(gen);
    data[i * 2] = x;
    data[i * 2 + 1] = dataFunction(x);
  }
  return data;
}

double lossFunction(double* data, double* nn_out, int data_idx){
  /*int size = sizeof(nn_out) / sizeof(nn_out[0]); 
  size = 1;
  printf("%d\n", size);
  double* loss = new double[2];
  printf("g\n");
  for (int i = 0; i < size; ++i){
    loss[i] = (nn_out[i] - data[data_idx + i]) * (nn_out[i] - data[data_idx + i]);
  }
  */
  double loss = (data[0] - nn_out[0]) * (data[0] - nn_out[0]);
  return loss;
}

int main(){

  double* tdata = createData(50000);

  printf("1\n");
  for (int i = 0; i < 50000; ++i){
    printf("%lf \n", tdata[i]);
  }
  Layer H1(4, 2);
  Layer H2(4, 4);
  Layer Out(1, 4);

  H1.forward(tdata);
  //printf("%lf \n", H1.f_output[0]);
  H1.relu(H1.f_output);
  //printf("%lf \n", H1.a_output[0]);

  H2.forward(H1.a_output);
  H2.relu(H2.f_output);

  Out.forward(H2.a_output);
  Out.relu(Out.f_output);
  printf("f\n");
  printf("%lf \n", Out.a_output[0]);
  //printf("%lf \n", Out.a_output[1]);
  double loss = lossFunction(tdata, Out.a_output, 0);
  printf("%lf \n", loss);
  //printf("%lf \n", loss[1]);
  
  delete[] tdata;
  //delete[] loss;
  return 0;
}
