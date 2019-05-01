"Differentiable Stack"
module NStack

# What do I want to happen 
# I want after the pops, to observe the value 
# Don't want to hardcode stacksixe in NStack
# on other hand we need to make an instantiation of the network
# which depends on the stack size
# 

using DSLearn
import DSLearn: isobservable, observe!
using DataStructures
using Flux
using FluxAddons

## Reference
"Empty stack of type `T`"
Base.empty(eltype::Type{T}, ::Type{Stack}) where T = Stack{T}()

## Neural
stack_size = (28, 28)

"Differentiable Stack"
mutable struct NStack{A <: AbstractArray}
  data::A
end
Base.size(::Type{Stack}) = stack_size

isobservable(::Any) = true
isobservable(::NStack) = false

"Empty stack"
Base.empty(eltype::Type, ::Type{NStack}; init = param ∘ zeros) =
  NStack(init(size(NStack)))

const push_net = mlp((Stack, Item), (Stack,))

function Base.push!(nstack::NStack, item)
  (stack,) = push_net(nstack, item)
  nstack.data = stack
  nstack
end

const pop_net = mlp((Stack,), (Stack, Item))

function Base.pop!(nstack::NStack)
  stack, item = pop_net(nstack)
  nstack.data = stack
  item
end

end