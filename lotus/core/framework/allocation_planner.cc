/*

Allocation Planner: goals, design choices, and initial plan:

The goal of the memory-sharing-planner is to analyze the lifetimes of tensors
and enable tensors whose lifetimes are disjoint to share memory. The goal is also
to do as much as possible statically to reduce inference-time. This planning is
complicated by the need to deal with "dynamic" tensors.

Explanation of terminology used: We use the term "statically" to denote things
that can be done when we load a model, and before any inputs for inference are
available. There are different categories of "dynamic" tensors, as explained below.
To begin with, we may categorize tensors into the following two categories:

(1) Statically named tensors, which are the primary kind of tensors we deal with.
Each such tensor has a static name, and the number of such tensors is statically
known (at the model definition time).

(2) Dynamically created tensors, which are not given a static name in the model.
The number of such tensors is not known statically. An example of a situation where
this can arise is if we have an operator that returns a "sequence of tensors" or
a "map from int to tensors". This is technically permitted by the ONNXML type-system.
I think we don't have such operators currently. Such tensors are out-of-scope for the
following discussion of static planning for memory-sharing among tensors.
(If such tensors become necessary, one way is to introduce reference-counted tensors
to realize these.)

We may further categorize statically named tensors into the following kinds:
(A) Tensors whose size is known statically (and does not vary from input to input).
(B) Tensors whose size is not known statically (and can vary from input to input).

For tensors of type A, we can do better memory-sharing. Further, we can do the
memory-allocation also statically, once for all.

For tensors of type B, the actual memory-allocation has to happen at inference time
when the size is actually known. However, we can do some planning statically.
In particular, we can partition the set of such tensors into several groups such
that all tensors in the same group can share the same memory. A necessary condition
for this grouping is that any two tensors in the same group must have disjoint lifetimes.

There are a few design choices here:

We can
(i) group together only tensors guaranteed to have the same (unknown) size,
e.g., two float tensors of shape (N,100,100), or
(ii) allow tensors of potentially different size, e.g., (N,100,100) and (M,50,50),
in the same group.
The second option permits more sharing, but may also result in some potentially
wasted space.

Note that the second option will require us to compute the maximum size of all tensors
in the group and allocate this much memory at memory-allocation time. We can also
try to make the grouping such that each group has as small a lifetime as possible.

The other important aspect of the planning algorithm is identifying the lifetimes
of tensors. This is easy in the absence of aliasing. However, we may wish to use
kernels that create aliasing (to reduce unnecessary copying): for example, operations
like reshape or slice can potentially return an output tensor that points to one of
the input tensor's data or to part of an input tensor's data.

To handle this we will require that the kernel-registration provide information
about the aliasing it can create. (We can use a default assumption of no-aliasing,
since that seems to be the most common.) Furthermore, we must distinguish between
must-aliasing and may-aliasing. A must-aliasing of input A and output B indicates
that the return value of B will definitely point to the data value of A (or a part
of A). This is straight-forward to use in the planner. A may-aliasing of input A
and output B indicates that the aliasing is conditional: it may be created in some
executions and not in other executions. The static-planner can treat such may-aliasing
conservatively, but that can result in some wastage in some situations.
The alternative is to use reference-counted tensors for the cases involving may-aliasing.

*/