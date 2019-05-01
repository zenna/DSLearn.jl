using DSLearn
import DSLearn: isobservable, observe!, net 
using FluxAddons: mlp
DSLearn.isobservable(::Any) = true

## Types
## =====
data(x) = x.data

"Learned representation of an Integer"
struct NInt{T} <: Integer
  data::T
end
NInt(x::Int) = convert(NInt, x)
Base.size(::Type{NInt}) = (3,)

"Continuous representation of an Integer"
struct CInt{T} <: Integer
  data::T
end
Base.size(::Type{CInt}) = (1,)

"Learned Bool"
struct NBool{T}
  data::T
end
Base.size(::Type{NBool}) = (1,)

struct CBool{T}
  data::T
end
Base.size(::Type{CBool}) = (1,)

## Conversions
## ===========
DSLearn.observe_type(::Type{<:NInt}) = CInt
DSLearn.observe_type(::Type{Int}) = CInt
DSLearn.observe_type(::Type{<:NBool}) = CBool
DSLearn.observe_type(::Type{Bool}) = CBool

Base.convert(T::Type{NInt}, x::CInt) = net(Base.convert, Type{NInt}, CInt)(x)
Base.convert(T::Type{NInt}, x::Int) = convert(NInt, convert(CInt, x))
Base.convert(T::Type{CInt}, x::NInt) = net(Base.convert, Type{CInt}, NInt)(x)

Base.convert(::Type{Bool}, x::NBool) = convert(Bool, convert(CBool, x))
Base.convert(::Type{CBool}, x::NBool) = net(Base.convert, Type{CBool}, NBool)(x)

## Interface
## =========
Base.:+(x::NInt, y::NInt) = net(+, NInt, NInt)(x, y)
Base.:+(x::NInt, y::Int) = x + convert(NInt, y) # FIXME DRY
Base.:-(x::NInt, y::NInt) = net(+, NInt, NInt)(x, y)
Base.:-(x::NInt, y::Int) = x - convert(NInt, y)
Base.:>(x::NInt, y::NInt) = net(>, NInt, NInt)(x, y)
Base.:>(x::NInt, y::Int) = x > convert(NInt, y)


Base.convert(::Type{CInt}, x::Int) = CInt(param([x])) #FIXME1
Base.convert(::Type{CBool}, x::Bool) = CBool(param([x]))
Base.convert(::Type{Bool}, x::CBool) = x.data[1] > 0.5

nmids = 5
nint_cint = mlp(CInt, NInt, nmids=[5])
@inline net(::typeof(Base.convert), ::Type{Type{NInt}}, ::Type{CInt}) = nint_cint

cint_nint = mlp(NInt, CInt, nmids=[5])
@inline net(::typeof(Base.convert), ::Type{Type{CInt}}, ::Type{NInt}) = cint_nint

cbool_nbool = mlp(NBool, CBool, nmids=[5])
@inline net(::typeof(Base.convert), ::Type{Type{CBool}}, ::Type{NBool}) = cbool_nbool

plus_net = mlp((NInt, NInt), NInt, nmids=[5])
@inline net(::typeof(+), ::Type{NInt}, ::Type{NInt}) = plus_net

sub_net = mlp((NInt, NInt), NInt, nmids=[5])
@inline net(::typeof(-), ::Type{NInt}, ::Type{NInt}) = sub_net

ngt_net = mlp((NInt, NInt), NBool, nmids=[5])
@inline net(::typeof(>), ::Type{NInt}, ::Type{NInt}) = ngt_net

function nets()
  return [ngt_net, sub_net, plus_net, cint_nint, nint_cint, cbool_nbool]
end

## Distances
## =========

δ(x::CInt, y::CInt) = mse(x.data, y.data)
δ(x::CBool, y::CBool) = mse(x.data, y.data)
