using DSLearn
import DSLearn: isobservable, observe!
using DataStructures
using MLDatasets
using PyTorch
using PyCall
import Base.Iterators
import IterTools
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
  s = empty(eltype(items), StackT)
  # Push n items
  for i = 1:nrounds
    @show i
    v = take!(items)
    @show v["size"]()
    push!(s, v)
  end

  # Pop n items
  for i = 1:nrounds
    i = observe!(pop!(s))
  end
  return s
end

"Infinite generator of batches from data"
function infinite_batches(data, batch_dim, batch_size, nelems = size(data, batch_dim))
  ids = Iterators.partition(cycle(1:nelems), batch_size)
  (slicedim(data, batch_dim, id) for id in ids)
end

function train_stack(batch_size = 128)
  train_x, _ = MNIST.traindata()
  train_x = permutedims(train_x, (3, 1, 2))
  batchgen_ = infinite_batches(train_x, 1, batch_size)
  batchgen = IterTools.imap(autograd.Variable ∘ PyTorch.torch.Tensor ∘ float, batchgen_)
  function producer(c::Channel)
    for x in batchgen
      put!(c, x)
    end
  end
  items1 = Channel(producer)
  items2 = Channel(producer)
  # Test with normal stack
  ref_ex1 = items -> ex1(items1, Stack)
  net_ex1 = items -> ex1(items2, NStack)
  println("Doing reference")
  trace1 = trace(ref_ex1, batchgen)
  println("Doing net")
  trace2 = trace(net_ex1, batchgen)
  trace1, trace2
end

# train_stack()