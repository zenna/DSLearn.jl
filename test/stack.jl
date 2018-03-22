using DSLearn
import DSLearn: isobservable, observe!
using DataStructures
using MLDatasets
using PyTorch
using PyCall
import Base.Iterators
import IterTools
@pyimport asl

struct Image{T}
  data::T
end

## Reference
"Empty stack of type `T`"
empty(eltype::Type, ::Type{Stack}) = Stack(eltype)

## Neural
stack_size = (28, 28)
item_size = (28, 28)
batch_size = 128

"Differentiable Stack"
mutable struct NStack{T}
  data::Tensor
end

isobservable(::Any) = true
isobservable(::NStack) = false

emptynstack = autograd.Variable(PyTorch.torch.randn(batch_size, stack_size...),
                                requires_grad = true)

"Empty stack"
empty(T::Type, ::Type{NStack}) = NStack{T}(emptynstack)

# Push
push_net = asl.MLPNet([stack_size, item_size], [stack_size])

function Base.push!(nstack::NStack, item)
  (item,) = push_net(nstack.data, item.data)
  NStack(nstack)
end

# Pop
pop_net = asl.MLPNet([stack_size], [stack_size, item_size])

function Base.pop!(nstack::NStack{T}) where T
  stack, item = pop_net(nstack.data)
  nstack.data = stack
  T(item)
end

## Test
## ====
"Example program"
function ex1(items, StackT::Type, nrounds=1, itemtype = eltype(items))
  s = empty(itemtype, StackT)
  # Push n items
  for i = 1:nrounds
    v = take!(items)
    push!(s, v)
  end

  # Pop n items
  for i = 1:nrounds
    i = observe!(Symbol(:o, 1), pop!(s))
  end
  return s
end

allparams = [collect(push_net[:parameters]());
             collect(pop_net[:parameters]());
             emptynstack]

adam = optim.Adam(allparams)

δ(x::Image{Tensor}, y::Image{Tensor}) = functional.mse_loss(x.data, y.data)

function train_stack(batch_size = 128)
  train_x, _ = MNIST.traindata()
  train_x = permutedims(train_x, (3, 1, 2))
  batchgen_ = DSLearn.infinite_batches(train_x, 1, batch_size)
  batchgen = IterTools.imap(Image ∘ autograd.Variable ∘ PyTorch.torch.Tensor ∘ float, batchgen_)
  function producer(c::Channel)
    for x in batchgen
      put!(c, x)
    end
  end
  items1 = Channel(producer)
  items2 = Channel(producer)
  # Test with normal stack
  ref_ex1 = items -> ex1(items1, Stack, 2, Image)
  net_ex1 = items -> ex1(items2, NStack, 2, Image)
  params = 
  for i = 1:1000
    # println("Doing reference")
    trace1 = trace(Image, ref_ex1, batchgen)
    # println("Doing net")
    trace2 = trace(Image, net_ex1, batchgen)
    losses = DSLearn.losses(trace2, trace1, δ)
    @show loss = losses[1]
    loss[:backward]()
    adam[:step]()
  end
end

train_stack()