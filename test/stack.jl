using DSLearn
import DSLearn: isobservable, observe!
using DataStructures
using MLDatasets
using PyTorch
using PyCall
@pyimport asl

Tensor = PyObject

## Reference
"Empty stack of type `T`"
empty(eltype::Type, ::Type{Stack}) = Stack(eltype)

## Neural
stack_size = (28, 28)
item_size = (28, 28)
batch_size = 128

"Differentiable Stack"
mutable struct NStack
  data::Tensor
end

isobservable(::Any) = true
isobservable(::NStack) = false

"Empty stack"
empty(eltype::Type, ::Type{NStack}) =
  NStack(autograd.Variable(PyTorch.torch.zeros(128, 28, 28)))

# Push
push_net = asl.MLPNet([stack_size, item_size], [stack_size])

function Base.push!(nstack::NStack, item)
  (item,) = push_net(nstack.data, item)
  NStack(nstack)
end

# Pop
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
  @show s = empty(eltype(items), StackT)
  # Push n items
  for i = 1:nrounds
    @show v = pop!(items)
    @show push!(s, v)
  end

  # Pop n items
  for i = 1:nrounds
    @show i = observe!(pop!(s))
  end
  return s
end

function train_stack()
  train_x, _ = MNIST.traindata()
  items = permutedims(train_x, (3, 2, 1))
  all_items = [items[1:128, :, :] for i = 1:10]

  # Test with normal stack
  ref_ex1 = items -> ex1(items, Stack)
  net_ex1 = items -> ex1(items, NStack)
  trace1 = trace(ref_ex1, all_items)
  trace2 = trace(net_ex1, all_items)
end

train_stack()