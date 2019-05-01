using DSLearn
import DSLearn: isobservable, observe!
using DataStructures
using MLDatasets
import Base.Iterators
import IterTools
using TensorboardX

## Types
## =====

struct Image{T}
  data::T
end
isobservable(::Image) = true
Base.size(::Type{Image}) = (28, 28)

"Differentiable Stack"
mutable struct NStack{T}
  data::Tensor
end
isobservable(::NStack) = false
Base.size(::Type{NStack}) = (28, 28)

## Interface
## ==========
emptynstack = autograd.Variable(PyTorch.torch.randn(128, size(NStack)...),
                                requires_grad = true)

"Empty stack"
NStack(::Type{T}) where T = NStack{T}(emptynstack)

push_net = asl.MLPNet([size(NStack), size(Image)], [size(NStack)])
function Base.push!(nstack::NStack{T}, item::T) where T
  (item,) = push_net(nstack.data, item.data)
  NStack(nstack)
end

pop_net = asl.MLPNet([size(NStack)], [size(NStack), size(Image)])
function Base.pop!(nstack::NStack{T}) where T
  stack, item = pop_net(nstack.data)
  nstack.data = stack
  T(item)
end

## Test
## ====
"Example program"
function ex1(items, Stack, nrounds=1, I::Type = eltype(items))
  s = Stack(I)
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

"Another Stack Example"
function ex2(items, Stack, nrounds=1, I::Type = eltype(items))
  s = Stack(I)
  push!(s, take!(items))
  observe!(:a, pop!(s))
  observe!(:b, isempty(s))
end


δ(x::Image{Tensor}, y::Image{Tensor}) = functional.mse_loss(x.data, y.data)

"Infinite iterator over MNIST"
function mnistcycle(batch_size)
  train_x, _ = MNIST.traindata()
  train_x = permutedims(train_x, (3, 1, 2))
  batchgen_ = DSLearn.infinite_batches(train_x, 1, batch_size)
  batchgen = IterTools.imap(Image ∘ autograd.Variable ∘ PyTorch.torch.Tensor ∘ float, batchgen_)
end

function tracegen(f, batchgen)
  function producer(c::Channel)
    i = 1
    for x in batchgen
      put!(c, x)
      @show i = i + 1
    end
  end
  items = Channel(producer)
  function tgen()
    trace1 = trace(Image, f, items)
  end
end

function train_stack(batch_size = 128)
  batchgen = mnistcycle(batch_size)
  ref_tracegen = tracegen(items -> ex1(items, Stack, 1, Image), batchgen)
  n_tracegen = tracegen(items -> ex1(items, NStack, 1, Image), batchgen)
  ref_tracegen, n_tracegen
end

ref_ds, n_ds = train_stack()

writer = SummaryWriter()

"""
Train Data Structure
"""
function step(ref_ds, n_ds, δ, fs, consts)
  params = vcat((collect(f[:parameters]()) for f in fs)...)
  allparams = [consts..., params...]
  adam = optim.Adam(allparams)
  function step!(cb_data, callbacks)
    ref_trace = ref_ds()
    net_trace = n_ds()
    losses = DSLearn.losses(net_trace, ref_trace, δ)
    loss = losses[1]
    add_image!(writer, "Empty", emptynstack[1], cb_data[:i])
    add_scalar!(writer, "Loss", loss, cb_data[:i])
    loss[:backward]()
    adam[:step]()
    @show loss
    loss
  end
end

step! = step(ref_ds, n_ds, δ, [push_net, pop_net], [emptynstack])

DSLearn.Optim.optimize(step!)