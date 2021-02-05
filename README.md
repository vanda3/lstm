# LSTM
LSTMs are a special kind of RNN, capable of learning long-term dependencies. LSTMs help preserve the error that can be backpropagated through time and layers. By maintaining a more constant error, they allow recurrent nets to continue to learn over many time steps.

Parameters being tested:
• n_epochs: number of times that the input is fed to the neural network 
• n_layers: number of Dense layers in the network
• n_units: number of hidden units in the LSTM
• layer_size: number of neurons in each Dense layer

# Run Code
python3.7 tensor.py
- Input file: gas.csv
