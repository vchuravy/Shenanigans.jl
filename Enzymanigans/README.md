# Enzyme.jl -- Prototyping

Goal: Allow Enzyme code AbstractInterpreter to recognize `autodiff` calls and to make it easier for them to infer.
Secondary goal: Allow for tape_type inference etc.

Use a custom abstract interpreter and take ideas from Diffractor.jl stage2.

We seperate entering the compiler plugin from the compiler intrinsics,
this allows us to write inference rules for our compiler intrinsics.

### Status:
1. Recognizes compiler intrinsic `autodiff`
2. return type inference of primal function


### TODO:
1. Mode and order. autodiff(x->autodiff)
2. Can we teach NativeInterpreter about CompilerPlugins and have custom_invoke be integrated in inference.
3. Actually integrate with Enzyme
  - Need an OpaqueClosure generator that embedds the LLVM IR from abicall


```
f(x) = Enzymanigans.autodiff(+, x, 1)
Enzymanigans.custom_invoke(f, 1)
```