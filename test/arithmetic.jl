## Types
## =====
using ArgParse
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

## Save / load from file
## Hyper parameter search
## 1. need to be able to save parameters to disk
## 2. 


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
  data::Tensor
end
# Base.promote_rule(::Type{NInt}, ::Type{Int}) = NInt
Base.size(::Type{NInt}) = (3,)

"Learned Bool"
struct NBool
  data::Tensor
end
Base.size(::Type{NBool}) = (1,)
PyCall.PyObject(x::NBool) = PyObject(x.data)

struct CBool
  data::Tensor
end
Base.size(CBool) = (1, )

## Conversions
## ===========
DSLearn.observe_type(::Type{NInt}) = CInt
DSLearn.observe_type(::Type{Int}) = CInt
DSLearn.observe_type(::Type{NBool}) = CBool
DSLearn.observe_type(::Type{Bool}) = CBool

Base.convert(::Type{NInt}, x::Int) =
  x |> variable |> net(convert, (Type{NInt}, Int)) |> first |> Tensor |> NInt

Base.convert(::Type{CInt}, x::NInt) = x |> net(Base.convert, Type{CInt}, NInt) |> first |> Tensor |> CInt
Base.convert(::Type{CInt}, x::Int) = CInt(variable(x))
Base.convert(::Type{CBool}, x::Bool) = CBool(variable(Int(x)))
Base.convert(::Type{Bool}, x::CBool) = x.data.data[:data][1,1] > 0.5
Base.convert(::Type{Bool}, x::NBool) = convert(Bool, convert(CBool, x))

Base.convert(::Type{CBool}, x::NBool) =
  x |> net(Base.convert, (Type{CBool}, NBool)) |> first |> functional.sigmoid |> Tensor |> CBool

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
  z = observe!(:b, x + rand(rng, 1:10))
  if convert(Bool, observe!(:b1, z > 5))
    z = observe!(:c1, z + rand(rng, 1:10))
  else
    z = observe!(:c2, z - rand(rng, 1:10))
  end
  z
end

## Distances
## =========
function mse(x, y)
  l = PyTorch.torch.norm(x.data - y.data)
end

δ(x::CInt, y::CInt) = mse(x.data, y.data)
δ(x::CBool, y::CBool) = mse(x.data, y.data)

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

function optimzer(fs, consts)
  params = vcat((collect(f[:parameters]()) for f in fs)...)
  allparams = [consts..., params...]
  optimizer = optim.Adam(allparams, lr = Param[:lr])
end

function stepgen(ref_ds, n_ds, δ, optimizer; traceperstep=64)
  function step!(cb_data, callbacks)
    all_losses = []
    for i = 1:traceperstep
      net_trace = n_ds()
      ref_trace = ref_ds()
      losses = DSLearn.losses(net_trace, ref_trace, δ)
      all_losses = vcat(all_losses, losses)
    end
    if cb_data[:i] % 100 == 0
      @show all_losses
    end
    @show loss = PyTorch.torch.stack(all_losses)[:mean]()
    add_scalar!(writer, "Loss", loss, cb_data[:i])
    loss[:backward]()
    optimizer[:step]()
    loss
    # @assert false
  end
end

function load_nets!(p::Param)
  nmids = get(p, :nmids, rand(1:10))
  net(Base.convert(::Type{NInt}, x::Int)) =
    asl.MLPNet([(1,)], [size(NInt)], batch_norm=false, nmids=PyVector([nmids]))
  
  net(::Type{CBool}, x::NBool) =
    asl.MLPNet([size(NBool)], [size(CBool)], batch_norm=false, nmids=PyVector([nmids]))

  
  net(::Type{CInt}, ::NInt) = 
    nint_cint = asl.MLPNet([size(NInt)], [size(CInt)], batch_norm=false, nmids=PyVector([nmids]))
end

function load_param(fn::String)
end

function train()
  p = load_param("arith_params.param")
  load_nets!(p)
  ref_tg, n_tg = train_arithmetic()
  optimizer = optimizer([nplus, nsub, ngt, n_nint, nint_cint, nbool_cbool], [])
  step! = stepgen(ref_tg, n_tg, δ, optimizer)
  save_params!()
  DSLearn.Optim.optimize(step!)
end

function runparams()
  s = ArgParseSettings()
  @add_arg_table s begin
    "--train"
        help = "Train the model"
    "--name", "-o"
        help = "Name of job"
        arg_type = Int
        default = 0
    "--log_dir"
        help = "Path to store data"
        action = :store_true
    "--resume_path"
        help = "Path to resume parameters from"
        required = true
    "--nocuda"
        help = "disables CUDA training"
        required = true
    "--dispatch"
        help = "disables may jobs"
        required = true
    "--optfile"
        help = "Specify load file to get options from"
        required = true
    "--nsamples"
        help = "number of samples for hyperparameters (default: 10)"
        required = true
    "--slurm"
        help = "number of samples for hyperparameters (default: 10)"
        required = true
    "--dryrun"
        help = "number of samples for hyperparameters (default: 10)"
        required = true
  end
end

#   parser.add_argument('--train', action='store_true', default=False,
#   help='Train the model')
# parser.add_argument('--name', type=str, default='', metavar='JN',
#     help='Name of job')
# parser.add_argument('--group', type=str, default='', metavar='JN',
#     help='Group name')
# parser.add_argument('--batch_size', type=int, default=16, metavar='N',
#     help='input batch size for training (default: 64)')
# parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
#     help='input batch size for testing (default: 1000)')
# parser.add_argument('--epochs', type=int, default=10, metavar='N',
#     help='number of epochs to train (default: 10)')
# parser.add_argument('--log_dir', type=str, metavar='D',
#     help='Path to store data')
# parser.add_argument('--resume_path', type=str, default=None, metavar='R',
#     help='Path to resume parameters from')
# parser.add_argument('--nocuda', action='store_true', default=False,
#     help='disables CUDA training')
# parser.add_argument('--seed', type=int, default=1, metavar='S',
#     help='random seed (default: 1)')
# parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
#     help='learning rate (default: 0.01)')

# def add_dispatch_args(parser):
# # Dispatch args
# parser.add_argument('--dispatch', action='store_true', default=False,
#     help='Dispatch many jobs')
# parser.add_argument('--sample', action='store_true', default=False,
#   help='Sample parameter values')
# parser.add_argument('--optfile', type=str, default=None,
#   help='Specify load file to get options from')
# parser.add_argument('--jobsinchunk', type=int, default=1, metavar='C',
#     help='Jobs to run per machine (default 1)')
# parser.add_argument('--nsamples', type=int, default=1, metavar='NS',
#     help='number of samples for hyperparameters (default: 10)')
# parser.add_argument('--blocking', action='store_true', default=True,
#     help='Is hyper parameter search blocking?')
# parser.add_argument('--slurm', action='store_true', default=False,
#     help='Use the SLURM batching system')
# parser.add_argument('--dryrun', action='store_true', default=False,
#   help='Do a dry run, does not call subprocess')
# end

# Problem is that we either have a global learning rate or pass around this param object
# Globals are generally bad but otherwise we need to keep passing around
# why is passing around bad? Because everyone in ebtween needs to know about this