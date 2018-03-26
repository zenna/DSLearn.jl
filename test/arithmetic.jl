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

## TODO
## New type for Continuous Representation
## Fix batch size mess.
## If there's control flow can only have batch_size of 1
## 

DSLearn.isobservable(::Any) = true

function variable(x::Int, requires_grad=false)
  x = fill(float(x), (1, 1))
  autograd.Variable(PyTorch.torch.Tensor(x), requires_grad=requires_grad)
end

## Types
## =====

"Learned representation of an Integer"
struct NInt <: Integer
  data::Tensor
end
NInt(x::Int) = convert(NInt, x)
PyCall.PyObject(x::NInt) = PyObject(x.data)

"CInt continuous representation of an Integer"
struct CInt <: Integer
  data::Tensor # TODO Type of this ::Float64
end
# Base.promote_rule(::Type{NInt}, ::Type{Int}) = NInt
Base.size(::Type{NInt}) = (3,)

"Learned Bool"
struct NBool{T}
  data::T
end
Base.size(::Type{NBool}) = (1,)
PyCall.PyObject(x::NBool) = PyObject(x.data)

struct CBool
  data # TODO: type of thisFloat64
end
Base.size(CBool) = (1, )

## Conversions
## ===========
DSLearn.observe_type(::Type{NInt}) = CInt
DSLearn.observe_type(::Type{Int}) = CInt
DSLearn.observe_type(::Type{NBool{T}}) where T <: Any = CBool
DSLearn.observe_type(::Type{Bool}) = CBool

n_nint = asl.MLPNet([(1,)], [size(NInt)], batch_norm=false, nmids=PyVector([5]))
Base.convert(::Type{NInt}, x::Int) =
  x |> variable |> n_nint |> first |> Tensor |> NInt

nint_cint = asl.MLPNet([size(NInt)], [size(CInt)], batch_norm=false, nmids=PyVector([5]))
Base.convert(::Type{CInt}, x::NInt) = x |> nint_cint |> first |> Tensor |> CInt

Base.convert(::Type{CInt}, x::Int) = CInt(variable(x))
Base.convert(::Type{CBool}, x::Bool) = CBool(variable(Int(x)))
Base.convert(::Type{Bool}, x::CBool) = x.data.data[:data][1,1] > 0.5
Base.convert(::Type{Bool}, x::NBool) = convert(Bool, convert(CBool, x))

nbool_cbool = asl.MLPNet([size(NBool)], [size(CBool)], batch_norm=false, nmids=PyVector([5]))
Base.convert(::Type{CBool}, x::NBool) =
  x |> nbool_cbool |> first |> functional.sigmoid |> Tensor |> CBool

## Interface
## =========
nplus = asl.MLPNet([size(NInt), size(NInt)], [size(NInt)], batch_norm=false, nmids=PyVector([5]))
Base.:+(x::NInt, y::NInt) = nplus(x, y) |> first |> Tensor |> NInt
Base.:+(x::NInt, y::Int) = x + NInt(y)

nsub = asl.MLPNet([size(NInt), size(NInt)], [size(NInt)], batch_norm=false, nmids=PyVector([5]))
Base.:-(x::NInt, y::NInt) = nsub(x, y) |> first |> Tensor |> NInt
Base.:-(x::NInt, y::Int) = x - NInt(y)

ngt = asl.MLPNet([size(NInt), size(NInt)], [size(NBool)], batch_norm=false, nmids=PyVector([5]))
Base.:>(x::NInt, y::Int) = ngt(x, NInt(y)) |> first |> Tensor |> NBool

## Programs
## ========
function ex1(rng, I::Type)
  x = observe!(:a, I(1) + I(2))
  z = observe!(:b, x + rand(rng, 1:100))
  if convert(Bool, observe!(:b1, z > 50))
    z = observe!(:c1, z + rand(rng, 1:100))
  else
    z = observe!(:c2, z - rand(rng, 1:100))
  end
  z
end

## Distances
## =========
function mse(x, y)
  # @show x, y  
  x = PyTorch.torch.stack([x, y], dim=-1)
  n = PyTorch.torch.norm(x, 2, -1)
  PyTorch.torch.mean(n)
end

δ(x::CInt, y::CInt) = mse(x.data, y.data)
δ(x::CBool, y::CBool) = mse(x.data, y.data)

# δ(x::Int, y::NInt) = δ(promote(x, y)...)
# δ(x::NInt, y::Int) = δ(promote(x, y)...)
# function δ(x::NInt, y::NInt)
#   x = PyTorch.torch.stack([x.data, y.data], dim=-1)
#   n = PyTorch.torch.norm(x, 2, -1)
#   PyTorch.torch.mean(n)
# end

# function δ(x::NBool, y::Bool)
#   y_ = variable(Int(true))
# end

## Training
## ========
writer = SummaryWriter()

function train_arithmetic(batch_size = 1)
  nrng = MersenneTwister(1234)
  refrng = MersenneTwister(1234)
  ref_tg() = trace(Any, () -> ex1(refrng, Int64))
  n_tg() = trace(Any, () -> ex1(nrng, NInt))
  ref_tg, n_tg
end

function stepgen(ref_ds, n_ds, δ, fs, consts; traceperstep=32)
  params = vcat((collect(f[:parameters]()) for f in fs)...)
  allparams = [consts..., params...]
  adam = optim.Adam(allparams)
  
  function step!(cb_data, callbacks)
    all_losses = []
    for i = 1:traceperstep
      # @show i
      # println/("Net")
      net_trace = n_ds()
      # println("Ref")
      ref_trace = ref_ds()
      losses = DSLearn.losses(net_trace, ref_trace, δ)
      all_losses = vcat(all_losses, losses)
    end
    @show loss = PyTorch.torch.stack(all_losses)[:mean]()
    add_scalar!(writer, "Loss", loss, cb_data[:i])
    loss[:backward]()
    adam[:step]()
    loss
    # @assert false
  end
end

ref_tg, n_tg = train_arithmetic()
step! = stepgen(ref_tg,
                n_tg,
                δ,
                [nplus, nsub, ngt, n_nint, nint_cint, nbool_cbool],
                [])
# step!(1, 2)
DSLearn.Optim.optimize(step!)