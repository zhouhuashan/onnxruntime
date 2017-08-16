#### Microsoft Confidential

Common IR
=========

The Common IR is an open specification that consists of the following
components:

1)  Definition of extensible computation graph model including methods
    for registering operators and manipulating the graph

2)  Built-in operators and standard data types

3)  Reference implementation of the built-in of operators

Extensible computation graph model
----------------------------------

The common IR specifies the portable, serialized format of the
computation graph. It may not be the form a framework chooses to use and
manipulate internally. For example, a framework may keep the graph in
memory in another format that it finds more efficient to manipulate for
optimization passes.

The computation dataflow graph is structured as a collection of nodes
that form a directed acyclic graph. Nodes have one or more inputs and
one or more outputs. Each node can be a built-in (intrinsic) operator or
a custom operator.

### Operators

See [Operators.md](Operators.md) for details

The IR also provides a method to query the set of available
implementations for any operator.

### Functions

Functions are subgraphs that encapsulate some functionality. They have
one or more inputs and one or more outputs. Functions are user-defined
and are analogous to subroutines in programming languages. They enable
composition of lower level operations into higher level ones.

Built-in operators and standard data types
------------------------------------------

### Standard data types

The following data types are supported by the Common IR. Additional data
types can be supported by frameworks.

-   float32

-   float64

-   uint8

-   int8

-   uint16

-   int16

-   int32

-   int64

-   string

-   bool

-   Tensor

Reference implementation
------------------------

An open source reference implementation of the built-in operators will
be provided. This can be used for testing custom implementations for
correctness.
