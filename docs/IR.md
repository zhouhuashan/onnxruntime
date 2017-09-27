#### Microsoft Confidential

Common IR
=========

The Common IR is an open specification that consists of the following components:

1)  Definition of an extensible computation graph model.

2)  Definition of built-in operators and standard data types.

3)  Reference implementations of graph management and operators.

__Some notes on language in this and all related documents__:

1. The use of SHOULD, MUST, MAY and so on in this document for normative purposes, which is indicated by upper-case lettering, is consistent with [RFC 2119](https://www.ietf.org/rfc/rfc2119.txt).

2. The use of 'list' shall denote an ordered collection of items, 'set' shall denote an unordered collection of unique elements, and 'bag' an unordered colletion of possibly non-unique elements.

Extensible computation graph model
----------------------------------

The common IR specifies the portable, serialized format of the computation graph. It may not be the form a framework chooses to use and manipulate internally. For example, a framework may keep the graph in memory in another format that it finds more efficient to manipulate for optimization passes.

### Graphs and Libraries

Each computation dataflow graph is structured as a list of nodes that form a graph, which MUST be free of cycles. Each node represents a call to an operator or a function. Each node SHOULD have zero or more inputs and one or more outputs. 

A serialized graph is comprised of a set of metadata fields, a set of model parameters, a list of computation nodes, a list of function definitions, and a list of operator declarations. Libraries are  comprised of metadata fields, a list of function definitions, and a list of operator declarations. A library does __not__ contain graphs outside of its function definitions.

Each graph and library MUST specify a name and a domain. Domains SHOULD be specified using reverse domain names as organization identifiers, the same convention that is used for naming Java packages.

Each graph MUST define the names and types of its inputs and outputs.

Graphs and libraries SHOULD be populated with documentation strings, which MAY be interpreted using markdown syntax. HTML and other text-markup languages MUST NOT be used in documentation strings.

__TODO: Define which specific markdown syntax to support.__

#### Metadata

The following are the required metadata properties of a model graph or library:

|Name|Type|Format|Description|
|----|----|------|-----------|
|name|string|Valid C identifier|A name for the model.|
|domain|string|Valid DNS name|A domain for the model, following the style of Java package names, that is, reverse domain name.|
|ir_version|int64||The version of the LotusIR specification.|
|model_version|int64||A version number of the model.|

The following are optional metadata properties of a model graph or library:

|Name|Type|Format|Description|
|----|----|------|-----------|
|model_author|string||The name of the author(s) of the model.|
|model_license|string||The name or URL defining the license under which the model is made available.|
|producer_tag|string||The name of the framework that produced the model.|
|producer_version|int64||The version of the framework that produced the model.|
|doc_string|string|Free form, markdown.|A human-readable documentation string intended to summarize the purpose and use of the model.|


#### Names Within a Graph

Names of nodes, inputs, outputs, initializers, attributes, and (in the case of graphs that are found within function definitions) parameters MUST be unique within that graph. Names of graph and library functions and operators share a namespace and MUST therefore be unique.  

All names MUST adhere to C identifier syntax rules.

#### Function Definitions

Functions are subgraphs that encapsulate some computation, which is expressed as a computation graph. Functions have one or more inputs and one or more outputs. Functions are user-defined and are analogous to subroutines in programming languages. They enable composition of lower level operations into higher level ones.

Function definitions are comprised of a name, a list of named input arguments, a list of named output arguments, a list of nodes, and a list of attributes.  

The inputs and outputs of a function are defined as lists of named, typed, formal parameters. The parameter names must be unique among the names within the function.

__OPEN QUESTION: What are function attributes used for? Do they allow passing values to functions? If so, how are they referenced within the function body?__

#### Nodes

Computation nodes are comprised of a name, a list of named inputs, a list of named outputs, a list of attributes, and a list of control inputs.

Edges in the computation graph are established by outputs of one node being referenced by name in the inputs of a subsequent node. Node, input, and output names MUST be unique within the graph. For a function definition, the names MUST be unique within the function definition's graph. Input nodes may also refer to declared graph inputs and graph initializers. Graph outputs refer to a subset of node outputs.

The list of input names may sometimes be longer than the list of parameters accepted by an operator that accepts more than one input value for a given parameter. To associate the input list with operator parameters, a separate list is used to define how many inputs are to be associated with each parameter. The inputs are matched with parameters left-to-right.

For example, an operator taking two arguments may be passed five values. In this example, an arg_count list '[2,3]' would mean that the first two inputs are associated with the first operator parameter, and the last three inputs with the second parameter. The input argument count list MUST be populated and its length MUST match the number of formal parameters of the operator.

__OPEN QUESTION: Are such arguments identified as sequences in operator declarations?__

Control inputs are used to establish edges in the computation graph that are based on other concerns than data dependencies. Inference runtime implementations MAY choose to ignore control edges when scheduling computations.

The list of nodes defining the top-level computation graph MUST be ordered topologically; that is, if node K follows node N in the graph, none of the data inputs of N may refer to outputs of K; further, no control input of N may refer to K.

Node attributes are used to pass literal (static) values, such as biases and other constants, to operators and functions.

#### Importing Libraries

Both graphs and libraries contain a list of imported libraries, which is the main mechanism for composing computation models out of more than one file. Libraries support sharing of common function definitions, as well as defining specific sets of operators that are supported by a runtime target. The root document loaded to perform a computation is always a graph, never a library.

Each library reference is in the form of a URI or relative path that represents the location of a document containing a serialized representation of the library. Tools MUST fetch all referenced documents, but MAY apply standard HTTP caching rules to avoid unnecessary fetches. Tools MAY bundle such library documents together with the root document (a graph), either by merging the library contents into the root document, or bundling them together through other mechanisms, thus freeing a runtime implementation from fetching library documents while loading a model for inference or training.

### Operators

All operators named within graph and library nodes MUST be explicitly declared within the graph or the transitive closure of its imported libraries. Such declarations MUST have full type information on all operands and SHOULD have shape information when applicable and possible. All operator attributes MUST also be typed. 

See [Operators.md](Operators.md) for details on the standard set of available operators.


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
|Collections|__map__|Maps represent associative tables, defined by a key type and a value type, both of which MUST NOT be a collection type.|

#### Tensor shapes

In addition to element type and dense/sparse properties, tensors have shape. A shape is a list of sizes that define whether the tensor is a vector, a matrix, or a higher-dimensioned structure. For example, a 100x100 matrix would have the shape [100,100].

The empty list of sizes, [], is a valid tensor shape. It's denotes a scalar value.

Each size in the list may be expressed as an integral value, or as a "dimension variable," a string denoting that the actual size of the dimension is not statically constrainted to a particular number, which is useful for declaring interfaces that care about the number of dimensions, but not the exact size of each dimension. For example, a NxM matrix would have the shape list [N,M].

Dimension variables are scoped to the declaration (graph signature, node signature, single operator declaration, or function definition signature) that they appear in. Thus, any given name denotes the same size within a declaration, allowing a declaration to describe how the shapes of inputs and outputs are related. For example, a function that performs matrix cross-product may be defined as taking two inputs of shape [K,M] and [M,N], and produce an output of shape [K,N].

Shapes MAY be defined using a combination of integers and variables.

__Wildcards__

Dimension variables MAY also include wildcards. A wildcard is used to denote a collection of sizes, or explicitly declare that no correlation is needed.

For example, the simple wildcard [\*\] denotes a tensor of any number of dimensions; using it on more than one declaration does not correlate sizes. Thus, decorating two inputs/outputs with [\*\] says nothing about how their respective shapes are related.

A wildcard may be _named_, which allows for correlation of shapes with wildcards.

Some wildcard examples:

```
[*]       
zero or more dimensions

[n,*]     
one or more dimensions
```

Correlating two shapes:

```
[m*],[m*]
The shape must be the same between both operands, but we don’t care exactly what it is.

[n,m*], [m*]
The second operand has one fewer dimensions, the left-most. The remaining dimensions are all the same as in the first operand.

[*,n], [n,*]   
We don’t care how many dimensions each operand has (other than “more than zero”), but the last of the left operand, and the first of the right operand, must have the same size. 
```
