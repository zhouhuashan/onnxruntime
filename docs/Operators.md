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

| Activation  | Math        | Tensor            | NN                   | RNN           |  Control   |  Logical       | Reduction       |
|-------------|-------------|-------------------|----------------------|-------------- |------------|----------------|-----------------|
| Sigmoid     | Exp         | Reshape           | Dropout              | RNNUnit       | While      | Greater        | ReduceMax       |
| Tanh        | Log         | Clip              | Pooling              | LSTMUnit      | If         | Less           | ReduceMin       |
| ReLU        | Sqrt        | Maximum           | Convolution          | GRUUnit       |            | Equal          | ReduceMean      |
| Softmax     | Floor       | Minimum           | ConvolutionTranspose |               |            | GreaterEqual   | ReduceSum       |
| ELU         | Abs         | Concatenate       | BatchNormalization   |               |            | LessEqual      | ReduceLogSumExp |
| LeakyRelu   | Reciprocal  | Slice             | ROIPooling           |               |            | And            | ReduceProd      |
| SoftPlus    | Plus        | Transpose         | Unpooling            |               |            | Or             |                 |
| PRelu       | Minus       | OneHot            |                      |               |            | Not            |                 |
| SELU        | Time        | ArgMax            |                      |               |            | NotEqual       |                 |
|             | Div         | ArgMin            |                      |               |            |                |                 |
|             | Dot         | Gather            |                      |               |            |                |                 |
|             | Pow         | RandomUniform     |                      |               |            |                |                 |
|             | Neg         | RandomNormal      |                      |               |            |                |                 |
|             | Sin         | Fill              |                      |               |            |                |                 |
|             | Cos         | RandomUniformLike |                      |               |            |                |                 |
|             | Square      | RandomNormalLike  |                      |               |            |                |                 |
|             | Sign        | FillLike          |                      |               |            |                |                 |
|             | Ceil        |                   |                      |               |            |                |                 |

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

`cos(x, name='')`

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

`reshape(x, shape, name='')`

Arguments | Description
----------| -----------
x         | tensor value or the result of an expression.
shape     | tuple of the new shape.
name      | Optional name.

#### Description:
Reshape the input tensor `x`.

### Clip

`clip(x, min, max, name='')`

Arguments | Description
----------| -----------
x         | tensor value or the result of an expression.
min       | min value.
max       | max value.
name      | Optional name.

#### Description:
Clip all tensor `x` value to be between `min` and `max`.

### Maximum

`maximum(left, right, name='')`

Arguments | Description
----------| -----------
left      | tensor value or the result of an expression.
right     | tensor value or the result of an expression.
name      | Optional name.

#### Description:
Computes the element-wise max of the two or more input tensors. It supports broadcasting.

### Minimum

`minimum(left, right, name='')`

Arguments | Description
----------| -----------
left      | tensor value or the result of an expression.
right     | tensor value or the result of an expression.
name      | Optional name.

#### Description:
Computes the element-wise min of the two or more input tensors. It supports broadcasting.

### Concatenate

`concatenate(inputs, axis, name='')`

Arguments | Description
----------| -----------
inputs    | One or more tensors.
axis      | The axis along which the concatenation occur.
name      | Optional name.

#### Description:
Concatenate the input tensors along an axis.

### Slice

`slice(x, begin, end, stride, name='')`

Arguments | Description
----------| -----------
x         | Tensor.
begin     | Shape tuple the start of slicing long each axis.
end       | Shape tuple the end of slicing long each axis.
stride    | Shape tuple the stride of slicing long each axis.
name      | Optional name.

#### Description:
Slice the input along one or multiple axes.

### Transpose

`transpose(x, perm, name='')`

Arguments | Description
----------| -----------
x         | Input tensor.
perm      | Tuple contain permutation applied to the axes.
name      | Optional name.

#### Description:
Permutes the axes of the tensor. The output has the same data but the axes are permuted according to `perm`.

### OneHot

`one_hot(x, num_classes, axis=-1, sparse_output=False, name='')`

Arguments     | Description
--------------| -----------
x             | Input tensor.
num_classes   | Number of the classes to generate the one hot presentation.
axis          | Axis along which to compute the one hot presentation.
sparse_output | If True, that the result will be stored in SparseTensor.
name          | Optional name.

#### Description:
Create one hot tensor based on the input tensor.

### ArgMax

`argmax(x, axis=None, name='')`

Arguments     | Description
--------------| -----------
x             | Input tensor.
axis          | Axis along which to compute the argmax.
name          | Optional name.

#### Description:
Computes the argmax of the input tensor’s elements across the specified axis. If no axis is specified, it will return the flatten index of the largest element in tensor x.

### ArgMin

`argmin(x, axis=None, name='')`

Arguments     | Description
--------------| -----------
x             | Input tensor.
axis          | Axis along which to compute the argmin.
name          | Optional name.

#### Description:
Computes the argmin of the input tensor’s elements across the specified axis. If no axis is specified, it will return the flatten index of the largest element in tensor x.

### Gather

`gather(x, indices, name='')`

Arguments     | Description
--------------| -----------
x             | Input tensor.
indices       | Indices of the element to return.
name          | Optional name.

#### Description:
Retrieves the elements of indices in the input tensor `x`.

### RandomUniform

`random_uniform(shape, low=0.0, high=1.0, seed=auto, dtype=float32, name='')`

Arguments     | Description
--------------| -----------
shape         | A tuple represent the shape of the return tensor.
low           | Lower end of the range of the random numbers.
high          | Upper end of the range of the random numbers.
seed          | Seed of the pseudo random generator or auto generate.
dtype         | The type of the returned tensor, default float32.
name          | Optional name.

#### Description:
Generates samples from the uniform distribution in the interval.

### RandomNormal

`random_normal(shape, mean=0.0, scale=1.0, seed=auto, dtype=float32, name='')`

Arguments     | Description
--------------| -----------
shape         | A tuple represent the shape of the return tensor.
mean          | Mean of the Gaussian distribution.
scale         | Standard deviation of the Gaussian distribution.
seed          | Seed of the pseudo random generator or auto generate.
dtype         | The type of the returned tensor, default float32.
name          | Optional name.

#### Description:
Generates samples from the normal distribution with mean and standard deviation scale.

### Fill

`fill(shape, value, dtype=float32, name='')`

Arguments     | Description
--------------| -----------
shape         | A tuple represent the shape of the return tensor.
value         | The value of the new tensor.
dtype         | The type of the returned tensor, default float32.
name          | Optional name.

#### Description:
Create a tensor with a specific value.

### RandomUniformLike

`random_uniform_like(x, low=0.0, high=1.0, seed=auto, dtype=float32, name='')`

Arguments     | Description
--------------| -----------
x             | Generate a random tensor with the same shape a `x`.
low           | Lower end of the range of the random numbers.
high          | Upper end of the range of the random numbers.
seed          | Seed of the pseudo random generator or auto generate.
dtype         | The type of the returned tensor, default float32.
name          | Optional name.

#### Description:
Generates samples from the uniform distribution in the interval.

### RandomNormalLike

`random_normal_like(x, mean=0.0, scale=1.0, seed=auto, dtype=float32, name='')`

Arguments     | Description
--------------| -----------
x             | Generate a random tensor with the same shape a `x`.
mean          | Mean of the Gaussian distribution.
scale         | Standard deviation of the Gaussian distribution.
seed          | Seed of the pseudo random generator or auto generate.
dtype         | The type of the returned tensor, default float32.
name          | Optional name.

#### Description:
Generates samples from the normal distribution with mean and standard deviation scale.

### FillLike

`fill_like(x, value, dtype=float32, name='')`

Arguments     | Description
--------------| -----------
x             | Generate a tensor with the same shape a `x`.
value         | The value of the new tensor.
dtype         | The type of the returned tensor, default float32.
name          | Optional name.

#### Description:
Create a tensor with a specific value.

## Logical Operators

### Greater

`greater(left, right, name='')`

Arguments | Description
----------| -----------
left      | tensor value or the result of an expression.
right     | tensor value or the result of an expression.
name      | Optional name.

#### Description:
Elementwise `greater` comparison of two tensors. Result is 1 if left > right else 0.

### Less

`less(left, right, name='')`

Arguments | Description
----------| -----------
left      | tensor value or the result of an expression.
right     | tensor value or the result of an expression.
name      | Optional name.

#### Description:
Elementwise `less` comparison of two tensors. Result is 1 if left < right else 0.

### Equal

`equal(left, right, name='')`

Arguments | Description
----------| -----------
left      | tensor value or the result of an expression.
right     | tensor value or the result of an expression.
name      | Optional name.

#### Description:
Elementwise `equal` comparison of two tensors. Result is 1 if left == right else 0.

### GreaterEqual

`greater_equal(left, right, name='')`

Arguments | Description
----------| -----------
left      | tensor value or the result of an expression.
right     | tensor value or the result of an expression.
name      | Optional name.

#### Description:
Elementwise `greater_equal` comparison of two tensors. Result is 1 if left >= right else 0.

### LessEqual

`less_equal(left, right, name='')`

Arguments | Description
----------| -----------
left      | tensor value or the result of an expression.
right     | tensor value or the result of an expression.
name      | Optional name.

#### Description:
Elementwise `less_equal` comparison of two tensors. Result is 1 if left <= right else 0.

### And

`and(left, right, name='')`

Arguments | Description
----------| -----------
left      | tensor value or the result of an expression.
right     | tensor value or the result of an expression.
name      | Optional name.

#### Description:
Elementwise `and` operator of two tensors. Result is 1 if left and right are 1 else 0.

### Or

`or(left, right, name='')`

Arguments | Description
----------| -----------
left      | tensor value or the result of an expression.
right     | tensor value or the result of an expression.
name      | Optional name.

#### Description:
Elementwise `or` operator of two tensors. Result is 1 if left or right are 1 else 0.

### Not

`not(x, name='')`

Arguments | Description
----------| -----------
x         | tensor value or the result of an expression.
name      | Optional name.

#### Description:
Elementwise `not` on the input tensor, flip 1 to 0 and vice versa.

### NotEqual

`not_equal(left, right, name='')`

Arguments | Description
----------| -----------
left      | tensor value or the result of an expression.
right     | tensor value or the result of an expression.
name      | Optional name.

#### Description:
Elementwise `not_equal` comparison of two tensors. Result is 1 if left != right else 0.

## Reduction Operators

### ReduceMax

`reduce_max(x, axes=None, keep_dims=True, name='')`

Arguments     | Description
--------------| -----------
x             | Input tensor.
axes          | Axes along which to compute the max.
keep_dims     | Keep the reduced dimension.
name          | Optional name.

#### Description:
Computes the max of the input tensor’s elements across a list of specified axes.

### ReduceMin

`reduce_min(x, axes=None, keep_dims=True, name='')`

Arguments     | Description
--------------| -----------
x             | Input tensor.
axes          | Axes along which to compute the min.
keep_dims     | Keep the reduced dimension.
name          | Optional name.

#### Description:
Computes the min of the input tensor’s elements across a list of specified axes.

### ReduceMean

`reduce_mean(x, axes=None, keep_dims=True, name='')`

Arguments     | Description
--------------| -----------
x             | Input tensor.
axes          | Axes along which to compute the mean.
keep_dims     | Keep the reduced dimension.
name          | Optional name.

#### Description:
Computes the mean of the input tensor’s elements across a list of specified axes.

### ReduceSum

`reduce_sum(x, axes=None, keep_dims=True, name='')`

Arguments     | Description
--------------| -----------
x             | Input tensor.
axes          | Axes along which to compute the sum.
keep_dims     | Keep the reduced dimension.
name          | Optional name.

#### Description:
Computes the sum of the input tensor’s elements across a list of specified axes.

### ReduceLogSumExp

`reduce_log_sum_exp(x, axes=None, keep_dims=True, name='')`

Arguments     | Description
--------------| -----------
x             | Input tensor.
axes          | Axes along which to compute the log of the sum of the exponentiations.
keep_dims     | Keep the reduced dimension.
name          | Optional name.

#### Description:
Computes the log of the sum of the exponentiations of the input tensor’s elements across a list of specified axes..

### ReduceProd

`reduce_prod(x, axes=None, keep_dims=True, name='')`

Arguments     | Description
--------------| -----------
x             | Input tensor.
axes          | Axes along which to compute the product.
keep_dims     | Keep the reduced dimension.
name          | Optional name.

#### Description:
Computes the product of the input tensor’s elements across a list of specified axes.

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
