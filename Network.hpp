#include "Layer.hpp"

class Network{
  int nlayers = 0;
  double loss = 0;
  int netInSize = 0;
  int netOutSize = 0;
  Layer** Layers;
  double * tdata; 
  double * tdesired_data;
  public: Network(Layer** layout, int num_layers);
  double squaredLoss(double*, double*, int);
  void forwardPass(double*);
  void backwardPass();
  void updateLayers(int samples, float learning_rate);
  public: void train(int epochs, int samples, double* data, double* desired_data, float learning_rate);
  public: void test(double*, double*, int);

  ~Network();
  
};
