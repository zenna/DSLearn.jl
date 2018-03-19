using DSLearn
using DataStructures
using MLDatasets
using PyTorch
using PyCall
@pyimport asl

## Reference
"Empty stack of type `T`"
empty(eltype::Type, ::Type{Stack}) = Stack(eltype)

## Neural
stack_size = (28, 28)
item_size = (28, 28)
batch_size = 128

"Differentiable Stack"
mutable struct NStack
  data
end

"Empty stack"
empty(eltype::Type, ::Type{NStack}) =
  NStack(autograd.Variable(PyTorch.torch.zeros(128, 28, 28)))

# Push
push_net = asl.MLPNet(PyObject([stack_size, item_size]), PyObject([stack_size]))
Base.push!(nstack::NStack, item) = ((item,) = push_net(nstack.data, item); NStack(nstack))
pop_net = asl.MLPNet(PyObject([stack_size]), PyObject([stack_size, item_size]))

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
    v = PyTorch.variable(pop!(items))
    push!(s, v)
  end

  # Pop n items
  for i = 1:nrounds
    i = pop!(s)
  end
  return s
end

function train_stack()
  train_x, _ = MNIST.traindata()
  items = permutedims(train_x, (3, 2, 1))
  all_items = [items[1:128, :, :] for i = 1:10]

  # Test with normal stack
  ex1(all_items, Stack, 1)

  # Test with neural stack
  ex1(all_items, NStack, 1)
end


function myfunc(f)
  while true
    okok
  end
  s = []
  push!(s, 1)
end