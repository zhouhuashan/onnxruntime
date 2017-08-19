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

Here the list of built-in operators that we plan to support:

| Activation  | Math        | Tensor         | NN                   | RNN           |  Control   |  Logical       | Reduction       |
|-------------|-------------|----------------|----------------------|-------------- |------------|----------------|-----------------|
| Sigmoid     | Exp         | Reshape        | Dropout              | RNNUnit       | While      | Greater        | ReduceMax       |
| Tanh        | Log         | Clip           | Pooling              | LSTMUnit      | If         | Less           | ReduceMin       |
| ReLU        | Sqrt        | Maximum        | Convolution          | GRUUnit       |            | Equal          | ReduceMean      |
| Softmax     | Floor       | Minimum        | ConvolutionTranspose |               |            | GreaterEqual   | ReduceSum       |
| ELU         | Abs         | Concatenate    | BatchNormalization   |               |            | LessEqual      | ReduceLogSumExp |
| LeakyRelu   | Reciprocal  | Slice          | ROIPooling           |               |            | And            | ReduceProd      |
| SoftPlus    | Plus        | Transpose      | Unpooling            |               |            | Or             |                 |
| PRelu       | Minus       | OneHot         |                      |               |            | Not            |                 |
| SELU        | Time        | ArgMax         |                      |               |            | NotEqual       |                 |
|             | Div         | ArgMin         |                      |               |            |                |                 |
|             | Dot         | Gather         |                      |               |            |                |                 |
|             | Pow         | Scatter        |                      |               |            |                |                 |
|             | Neg         | UniformRandom  |                      |               |            |                |                 |
|             | Sin         | NormalRandom   |                      |               |            |                |                 |
|             | Cos         | Fill           |                      |               |            |                |                 |
|             | Square      |                |                      |               |            |                |                 |
|             | Sign        |                |                      |               |            |                |                 |
|             | Ceil        |                |                      |               |            |                |                 |

## Control Flow Operators

### While

`while(condition, body, name='')`

Arguments          | Description
-------------------| -------------------------------
condition          | Stopping condition of the loop.
body               | Callable body of the loop.
name               | Optional name.

### If

`if(pred, true_function, false_function, name='')`

Arguments          | Description
-------------------| -----------
pred               | If pred !=0 then return true_function, otherwise return false_function.
true_function      | Value returned when pred != 0.
false_function     | Value returned when pred == 0.
name               | Optional name.

#### Description:

`pred` can be scalar or tensor. For scalar depend on its value, it will return `true_function` or `false_function`. However, for tensor the `pred` shape need to be the same as the return of `true_function` and `false_function`, and the evaluation will be done elementwise.

## Activation Operators

### Sigmoid

`sigmoid(x, name='')`

Arguments | Description
----------| -----------
x         | tensor value or the result of an expression.
name      | Optional name.

#### Description:
Computes the element-wise sigmoid of `x`.

### Tanh

`tanh(x, name='')`

Arguments | Description
----------| -----------
x         | tensor value or the result of an expression.
name      | Optional name.

#### Description:
Computes the element-wise tanh of `x`.

### ReLU

`relu(x, name='')`

Arguments | Description
----------| -----------
x         | tensor value or the result of an expression.
name      | Optional name.

#### Description:
Computes the element-wise relu of `x`.

### Softmax

`softmax(x, axis=None, name='')`

Arguments | Description
----------| -----------
x         | tensor value or the result of an expression.
axis      | If given, it will run softmax along that axis. Otherwise, softmax will be applied on all axis.
name      | Optional name.

#### Description:
Computes the `softmax` of `x` along axis. If axis=-1, it will compute `softmax` along the last axis. If axis=None, it will compute `softmax` along all axes.

### ELU

`elu(x, alpha=1, name='')`

Arguments | Description
----------| -----------
x         | tensor value or the result of an expression.
alpha     | Scalar constant default to 1.
name      | Optional name.

#### Description:
Computes the element-wise elu of `x`.

### LeakyRelu

`leaky_relu(x, name='')`

Arguments | Description
----------| -----------
x         | tensor value or the result of an expression.
name      | Optional name.

#### Description:
Computes the element-wise leaky relu of `x`.

### SoftPlus

`softplus(x, name='')`

Arguments | Description
----------| -----------
x         | tensor value or the result of an expression.
name      | Optional name.

#### Description:
Computes the element-wise softplus of `x`.

### PRelu

`prelu(x, alpha, name='')`

Arguments | Description
----------| -----------
x         | tensor value or the result of an expression.
alpha     | Constant or tensor.
name      | Optional name.

#### Description:
Computes the element-wise parametric relu of `x`. If alpha is a non-constant tensor, then it is learnable during training.

### SELU

`selu(x, name='')`

Arguments | Description
----------| -----------
x         | tensor value or the result of an expression.
name      | Optional name.

#### Description:
Computes the element-wise scaled exponential linear unit of `x`.

## Math Operators

### Exp

`exp(x, name='')`

Arguments | Description
----------| -----------
x         | tensor value or the result of an expression.
name      | Optional name.

#### Description:
Computes the element-wise exponential of `x`.

### Log

`log(x, name='')`

Arguments | Description
----------| -----------
x         | tensor value or the result of an expression.
name      | Optional name.

#### Description:
Computes the element-wise natural logarithm of `x`.

### Sqrt

`sqrt(x, name='')`

Arguments | Description
----------| -----------
x         | tensor value or the result of an expression.
name      | Optional name.

#### Description:
Computes the element-wise square root of `x`. Will return NaN for negative `x`.

### Floor

`floor(x, name='')`

Arguments | Description
----------| -----------
x         | tensor value or the result of an expression.
name      | Optional name.

#### Description:
Computes the element-wise floor of `x`.

### Abs

`abs(x, name='')`

Arguments | Description
----------| -----------
x         | tensor value or the result of an expression.
name      | Optional name.

#### Description:
Computes the element-wise absolute of `x`.

### Reciprocal

`reciprocal(x, name='')`

Arguments | Description
----------| -----------
x         | tensor value or the result of an expression.
name      | Optional name.

#### Description:
Computes the element-wise reciprocal of `x`.

### Plus

`plus(left, right, name='')`

Arguments | Description
----------| -----------
left      | tensor value or the result of an expression.
right     | tensor value or the result of an expression.
name      | Optional name.

#### Description:
Computes the sum of the `left` and `right` input tensors. It supports broadcasting.

### Minus

`minus(left, right, name='')`

Arguments | Description
----------| -----------
left      | tensor value or the result of an expression.
right     | tensor value or the result of an expression.
name      | Optional name.

#### Description:
Computes the difference of the `left` and `right` input tensors. It supports broadcasting.

### Time

`time(left, right, name='')`

Arguments | Description
----------| -----------
left      | tensor value or the result of an expression.
right     | tensor value or the result of an expression.
name      | Optional name.

#### Description:
Computes the product of the `left` and `right` input tensors. It supports broadcasting.

### Div

`div(left, right, name='')`

Arguments | Description
----------| -----------
left      | tensor value or the result of an expression.
right     | tensor value or the result of an expression.
name      | Optional name.

#### Description:
Computes the division of the `left` and `right` input tensors. It supports broadcasting.

### Dot

`dot(left, right, reduction_rank=1, name='')`

Arguments      | Description
---------------| -----------
left           | tensor value or the result of an expression.
right          | tensor value or the result of an expression.
reduction_rank | Represents the number of axes to be collapsed in order to transform the tensors into matrices, perform the operation and then reshape back.
name           | Optional name.

#### Description:
Computes the dot product of the `left` and `right` input tensors. It supports broadcasting.

### Pow

`pow(x, e, name='')`

Arguments | Description
----------| -----------
x         | base tensor.
e         | exponent tensor.
name      | Optional name.

#### Description:
Computes base raised to the power of exponent. It supports broadcasting. This is well defined if base is non-negative or exponent is an integer. Otherwise the result is NaN.

### Neg

`neg(x, name='')`

Arguments | Description
----------| -----------
x         | tensor value or the result of an expression.
name      | Optional name.

#### Description:
Computes the element-wise negative of `x`.

### Sin

`sin(x, name='')`

Arguments | Description
----------| -----------
x         | tensor value or the result of an expression.
name      | Optional name.

#### Description:
Computes the element-wise sin of `x`.

### Cos

`Cos(x, name='')`

Arguments | Description
----------| -----------
x         | tensor value or the result of an expression.
name      | Optional name.

#### Description:
Computes the element-wise cos of `x`.

### Square

`square(x, name='')`

Arguments | Description
----------| -----------
x         | tensor value or the result of an expression.
name      | Optional name.

#### Description:
Computes the element-wise square of `x`.

### Sign

`sign(x, name='')`

Arguments | Description
----------| -----------
x         | tensor value or the result of an expression.
name      | Optional name.

#### Description:
Return -1 if x < 0; 0 if x == 0 or NAN; 1 if x > 0.

### Ceil

`ceil(x, name='')`

Arguments | Description
----------| -----------
x         | tensor value or the result of an expression.
name      | Optional name.

#### Description:
Computes the element-wise ceil of `x`.


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
