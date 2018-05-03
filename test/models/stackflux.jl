using DSLearn
import DSLearn: isobservable, observe!
using DataStructures
using MLDatasets
using Flux
using Base.Iterators
using TensorboardX

## Reference
"Empty stack of type `T`"
empty(eltype::Type, ::Type{Stack}) = Stack(eltype)

## Neural
stack_size = (28, 28)
item_size = (28, 28)
batch_size = 128

"Differentiable Stack"
mutable struct NStack
  data::AbstractArray
end

isobservable(::Any) = true
isobservable(::NStack) = false

"Empty stack"
empty(eltype::Type, ::Type{NStack}) =
  NStack(param(zero(128, 28, 28)))

# Push
push_net = asl.MLPNet([stack_size, item_size], [stack_size])

function Base.push!(nstack::NStack, item)
  (item,) = push_net(nstack.data, item)
  NStack(nstack)
end

# Pop
cat ∘

pop_net = asl.MLPNet([stack_size], [stack_size, item_size])

function Base.pop!(nstack::NStack)
  stack, item = pop_net(nstack.data)
  nstack.data = stack
  item
end

## Test
## ====
"Example program"
function ex1(items, StackT::Type, nrounds=1)
  s = empty(eltype(items), StackT)
  
  # Push n items
  for i = 1:nrounds
    v = pop!(items)
    push!(s, v)
  end

  # Pop n items
  for i = 1:nrounds
    i = observe!(pop!(s))
  end
  return s
end


function train_stack()
  train_x, _ = MNIST.traindata()
  nelems = size(train_x, 3)
  batch_size = 128
  ids = partition(cycle(1:nelems), batch_size)
  ref_data = (train_x[:, :, id] for id in ids)

  # Test with normal stack
  ref_ex1 = items -> ex1(items, Stack)
  net_ex1 = items -> ex1(items, NStack)
  trace1 = trace(ref_ex1, all_items)
  trace2 = trace(net_ex1, all_items)
end