## Types
## =====
using DSLearn
import DSLearn: isobservable, observe!
using DataStructures
using MLDatasets
using PyTorch
using PyCall
import Base.Iterators
import IterTools
using TensorboardX
@pyimport asl

DSLearn.isobservable(::Any) = true

struct NInt{T} <: Integer
  data::T
end
Base.size(::Type{NInt}) = (64,)

nint_int = asl.MLPNet([(1,)], [size(NInt)])
Base.convert(::Type{NInt}, x::Int) = nint_int(x)

"Learned Bool"
mutable struct NBool{T}
  data::Tensor
end
Base.size(::Type{NBool}) = (1,)

nplus = asl.MLPNet([size(NInt), size(NInt)], [size(NInt)])
function Base.:+(x::NInt, y::NInt)
  (i,) = nplus(x.data, y.data)
  NInt(i)
end

nsub = asl.MLPNet([size(NInt), size(NInt)], [size(NInt)])
function Base.:-(x::NInt, y::NInt)
  (i,) = nsub(x.data, y.data)
  NInt(i)
end

ngt = asl.MLPNet([size(NInt), size(NInt)], [size(NBool)])
function Base.:>(x::NInt, y::Int)
  (i,) = ngt(x.data, y.data)
  NInt(i)
end

nsub = asl.MLPNet([size(NInt), size(NInt)], [size(NInt)])
function Base.convert(::Type{Bool}, ::NBool)
  NBool
end

function ex1(rng, I::Type)
  x = observe!(:a, I(1) + I(2))
  z = observe!(:b, x + rand(rng, 1:100))
  if Bool(observe!(:b1, z > 2))
    z = observe!(:c, z + 50)
  else
    z = observe!(:c, z - rand(rng, 1:100))
  end
  z
end

function train_arithmetic(batch_size = 1)
  nrng = MersenneTwister(1234)
  refrng = MersenneTwister(1234)
  ref_tg() = trace(Any, () -> ex1(nrng, Int64))
  n_tg() = trace(Any, () -> ex1(nrng, NInt))
  ref_tg, n_tg
end

writer = SummaryWriter()

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

δ(x::Int, y::NInt) = @assert false

ref_tg, n_tg = train_arithmetic()

step! = step(ref_tg,
             n_tg,
             δ,
             [nplus, nsub, ngt],
             [])
step!(1, 2)
# DSLearn.Optim.optimize(step!)