# Neural Network Example

This project provides a simple implementation of a feedforward neural network using C++. It consists of classes for the neural network (`Network`) and individual layers (`Layer`), along with a `main.cpp` file for demonstration.

## Prerequisites

- C++ compiler supporting C++11 or higher

## Getting Started

1. Clone the repository or download the source code files.
2. Ensure that the necessary header files (`Network.hpp` and `Layer.hpp`) are included in your project.
3. Build and compile the project using your preferred C++ compiler.

## Usage

The `main.cpp` file demonstrates an example usage of the neural network. It generates training data, specifies the network layout, and trains the network using the generated data.

1. Modify the `dataFunction` in `main.cpp` to define your desired mapping function or uncomment the provided function.
2. Adjust the parameters in `main.cpp` to suit your specific problem:
   - `LEARNING_RATE`: The learning rate for weight updates during training.
   - `SAMPLES`: The number of training samples to generate.
   - `EPOCHS`: The number of training epochs.
   - `layout`: The layout of the neural network, specifying the number of neurons in each layer.
3. Compile and run the program to train the neural network and observe the loss at each epoch.

## Customization

Feel free to customize and extend the code as needed for your specific use case. You can modify the network layout, activation functions, regularization techniques, and other parameters to suit your requirements.

## Contributing

Contributions to this project are welcome. If you encounter any issues or have suggestions for improvements, please submit an issue or a pull request.


## Acknowledgements

- The implementation of the neural network is inspired by various online resources and tutorials.

