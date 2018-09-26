#### Microsoft Confidential

FAQ
===

How can operator implementations from one framework be used with others since they likely depend on framework-specific memory managers, etc.?
---------------------------------------------------------------------------------------------------------------------------------------------

The Common IR specifies the set of operators all participating frameworks support. The Common IR does not imply that operator implementations are interchangeable across
frameworks. Each framework provides its own implementation of operators and these implementations may be optimized differently.

Can a framework choose to support an operator only on a subset of the standard data types?
------------------------------------------------------------------------------------------

Yes.
