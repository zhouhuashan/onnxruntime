#### Microsoft Confidential

Common IR
=========

The Common IR is an open specification that consists of the following
components:

1)  Definition of an extensible computation graph model.

2)  Built-in operators and standard data types.

3)  Reference implementation of the built-in of operators.

__Some notes on language in this and all related documents__:

1. The use of SHOULD, MUST, MAY and so on in this document is consistent with [RFC 2119](https://www.ietf.org/rfc/rfc2119.txt).

2. The use of 'list' shall denote an ordered collection of items, 'set' shall denote an unordered collection of unique elements, and 'bag' an unordered colletion of possibly non-unique elements.

Extensible computation graph model
----------------------------------

The common IR specifies the portable, serialized format of the computation graph. It may not be the form a framework chooses to use and manipulate internally. For example, a framework may keep the graph in memory in another format that it finds more efficient to manipulate for optimization passes.

### Graphs

Each computation dataflow graph is structured as a list of nodes that form a graph, which MUST be free of cycles. Nodes have one or more inputs and one or more outputs. Each node can be a call to a built-in (intrinsic) operator, a custom operator, or a function.

A serialized graph is comprised of a set of metadata fields, a set of model parameters, a list of computation nodes, and a list of function definitions.

#### Metadata

The following are the required metadata properties of a model graph:

|Name|Type|Format|Description|
|----|----|------|-----------|
|name|string|Valid C identifier|A name for the model.|
|namespace|string|Valid DNS name|A namespace for the model, following the style of Java package names, that is, reverse DNS domain name.|
|version|int64||A version number of the model|
|ir_version|string||The version of the IR format specification|
|documentation|string|Free form|A human-readable documentation string intended to summarize the purpose of the model.|

All optional metadata are organized in a <string,string> map. The set of optional metadata elements is extensible without revising the specification, but tools and runtime implementations are expected to understand the following optional metadata elements:

|Name|Description|
|----|----|
|author|The name of the individual or individuals that developed the model.|
|license|The name or URL defining the license under which the model is made available.|
|training_parameters|Parameters used when training this model in a framework-specific form.|
|training_dataset|Identifier of the dataset(s) used to train this model.|

#### Model Parameters

__TODO: Address the syntax and semantics for how model parameters are defined.__

#### Names Within a Graph

Names of nodes, inputs, outputs, and (in the case of graphs that are found within function definitions) parameters MUST be unique within that graph. Names of functions may overlap with other names, but a function MUST NOT have a name that can also be the name of an operator. 

All names MUST adhere to C identifier syntax rules.

#### Function Definitions

Functions are subgraphs that encapsulate some computation, which is expressed as a computation graph. Functions have one or more inputs and one or more outputs. Functions are user-defined and are analogous to subroutines in programming languages. They enable composition of lower level operations into higher level ones.

Function definitions are comprised of a name, a list of named input arguments, a list of named output arguments, a list of nodes, and a list of attributes.  

The inputs and outputs of a function are defined as lists of named, typed, formal parameters. The parameter names must be unique among the names within the function.

#### Nodes

Computation nodes are comprised of a name, a list of named inputs, a list of named outputs, a list of attributes, a list of control inputs, and a list of attributes.

Edges in the computation graph are established by outputs of one node being referenced by name in the inputs of a subsequent node. Node, input, and output names must be unique within the graph. For a function definition, the names must be unique within the function definition's graph.

The control inputs are used to establish edges in the computation graph that are based on other concerns than data dependencies. 

The list of nodes defining the top-level computation graph MUST be ordered topologically; that is, if node K follows node N in the graph, none of the data inputs of N may refer to outputs of K; further, no control input of N may refer to K.


__TODO: Address how the overall input/output signature of the graph is established. Function definitions have input/output formal argument definitions, but
that is not so for graphs. Is it from the first and last nodes in the graph? If so, does that mean that one node (the first) will dominate all others in the graph, and that one node (the last) will post-dominate all other nodes?__

__TODO: Describe how model parameters are referenced in nodes.__



### Operators

See [Operators.md](Operators.md) for details


Built-in operators and standard data types
------------------------------------------

### Standard data types

The following data types are supported by the Common IR. Additional data types can be supported by frameworks.

|Group|Name|Description|
|-----|----|-----------|
|Floating Point Types|__float16, float32, float64__|Values adhering to the IEEE 754-2008 standard representation of floating-point data.|
|Signed Integer Types|__int8, int16,int32,int64__|Signed integers are supported for 8-64 bit widths.|
|Unsigned Integer Types|__uint8,uint16__| Unsigned integers of 8 or 16 bits are supported.|
|Complex Types|__complex64,complex128__|A complex number with either 32- or 64-bit real and imaginary parts.|
|Other|__string__|Strings represent textual data. All strings are encoded using UTF-8.|
|Ohter|__bool__|Boolean value represent data with only two values, typically _true_ and _false_.|
|Other|__handle__|Handles are opaque types holding a 64-bit integer.|
|Collections|__sparse and dense tensor__|Tensors are a generalization of vectors and matrices; whereas vectors have one dimension, and matrices two, tensors can have any number of dimenstions, including zero. A zero-dimensional tensor is equivalent to a scalar.|
|Collections|__list__|Lists represent dense, ordered, collections of elements that are of homogeneous types. List elements can be added to the tail, removed from the head, and accessed by integer index.|
|Collections|__tuple__|Tuples represent dense, ordered, collections of elements of heterogeneous types. Tuple elements are accessed by integer index.|

__TODO: Add maps when they are added to the .proto file.__


Reference implementation
------------------------

An open source reference implementation of the built-in operators will be provided. This can be used for validating the correctness of custom implementations.
