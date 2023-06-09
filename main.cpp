#include "Layer.hpp"
#include <iostream>

int main(){
  double* data = new double[2];
  data[0] = 4, data[1] = 3;

  //printf("1\n");
  Layer H1(4, 2);
  Layer H2(4, 4);
  Layer Out(2, 4);

  H1.forward(data);
  //printf("%lf \n", H1.f_output[0]);
  H1.relu(H1.f_output);
  //printf("%lf \n", H1.a_output[0]);

  H2.forward(H1.a_output);
  H2.relu(H2.f_output);

  Out.forward(H2.a_output);
  Out.relu(Out.f_output);
  //printf("%lf \n", Out.a_output[0]);
  //printf("%lf \n", Out.a_output[1]);
}
