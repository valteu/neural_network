#include "Layer.hpp"

class Network{
  int nlayers = 0;
  double loss = 0;
  int netInSize = 0;
  int netOutSize = 0;
  Layer** Layers;
  double * tdata; 
  double * tdesired_data;
  Network(int* layout, int num_layers);
  double lossFunction(double*, double*, int);
  void forwardPath();
  void backwardPath();
  void updateLayers(int samples, float learning_rate);
  void train(int epochs, int samples, double* data, double* desired_data, float learning_rate);

  ~Network();
  
};
