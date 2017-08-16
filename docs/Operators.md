#### Microsoft Confidential

Operators
=========

Operators are implemented external to the graph and registered with the
IR. The IR provides a single registration mechanism for all operators. The
registration process maps operators to their implementation for specific
data types and on specific devices (CPU, GPU, FPGA, etc). This way the
IR cleanly represents the logical operation being performed while the
implementation is kept separate. Built-in operators are pre-registered
by the framework and custom operators can be registered by the framework
or by users.

Built-in operators are portable across frameworks while custom
operators may be framework-specific. Every framework adopting
the Common IR will provide implementations of these on all the standard
data types. However frameworks can chose which devices each operator
supports.

## Control Flow Operators

### While

`While(condition_function, body_function)`

Arguments | Description
--------- | -----------
condition_function | 
body_function | 

### Cond

`Cond(condition_function, true_function, false_function)`

Arguments | Description
--------- | -----------
condition_function | 
true_function | 
false_function | 

## Activation Operators

### Sigmoid
### Tanh
### ReLU
### Softmax
### ELU
### LeakyRelu
### SoftPlus
  
## Math Operators

### Exp
### Log
### Sqrt
### Floor
### Abs
### Reciprocal
### Plus

## Tensor Operators

### Reshape

`Reshape(condition_function, true_function, false_function)`

### Clip
### Maximum
### Minimum
### Concatenate
### Slice
### Transpose
### Minus
### Time
### Div
### Dot
### Pow

## NN Operators

### Dropout
### Pooling
### Convolution
### ConvolutionTranspose
### BatchNormalization
### ROIPooling
### Unpooling

## RNN Operators

### SimpleRNN
### LSTM
### GRU
