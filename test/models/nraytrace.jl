using Flux
using DSLearn
import RayTrace: Geometry, Scene, Vec3, Ray, Intersection, rgbimg
import FluxAddons: mlp
import ImageView: imshow
using RunTools

"Learned Geometry"
abstract type NGeometry <: Geometry end

"Learned sphere"
struct NSphere{T} <: NGeometry
  data::T
end

Base.size(::Type{NSphere}) = (4,) #TODO: parameterize this
Base.size(::Type{Vec3}) = (3,)

struct Vec3x{T}
  data::T
end
Base.size(::Type{Vec3x}) = (3,)

struct Scalar{T}
  data::T
end
Base.size(::Type{Scalar}) = (1,)

"Continuous Ray tyoe"
struct CRay{T}
  data::T
end
Base.size(::Type{CRay}) = (6,)

"Continuous Intersection type"
struct CIntersection{T}
  data::T
end
Base.size(::Type{CIntersection}) = (3, )

## Conversions
## ===========
DSLearn.observe_type(::Type{<:Intersection}) = CIntersection
DSLearn.observe_type(::Type{Ray}) = CRay
# DSLearn.observe_type(::Type{<:NBool}) = CBool
# DSLearn.observe_type(::Type{Bool}) = CBool

Base.convert(::Type{CRay}, ray::Ray) = CRay(addims(vcat(ray.orig, ray.dir)))
function Base.convert(::Type{Intersection}, cint::CIntersection)
  @grab cint
  @assert false
  Intersection(cint.data...)
end
function Base.convert(::Type{CIntersection}, int::Intersection)
  @assert false
end

addims(x) = reshape(x, size(x)..., 1)

## Networks
## ========

const NSphere_net = first ∘ mlp((Vec3, Scalar), (NSphere,)) # FIXME: doesn't work
NSphere(xyz::Vec3x, r) = NSphere_net(xyz, r) # FIXME Parameterie
NSphere(xyz::Vec3, r::Real) = NSphere(Vec3x(reshape(xyz, (3, 1))), Scalar(fill(r, (1, 1))))

const rayintersect_net = mlp((CRay, NSphere), (CIntersection))
RayTrace.rayintersect(r::CRay, s::NSphere) = rayintersect_net(r, s)  # FIXME Parameterie
function RayTrace.rayintersect(ray::Ray, nsphere::NSphere)
  cintersect = RayTrace.rayintersect(convert(CRay, ray), nsphere)
  convert(Intersection, cintersect)
end

## Test
## ====
"Some example spheres which should create actual image"
function example_spheres(SphereT::T) where T
  RayTrace.ListScene(
   [SphereT(Float64[0.0, -10004, -20], 10000.0),
    SphereT(Float64[0.0,      0, -20],     4.0),
    SphereT(Float64[5.0,     -1, -15],     2.0),
    SphereT(Float64[5.0,      0, -25],     3.0),
    SphereT(Float64[-5.5,     0, -15],     3.0),
    # light (emission > 0)
    SphereT(Float64[0.0,     20.0, -30],  3.0)])
end

function renderscene(SphereT::T) where T
  scene = example_spheres(SphereT)
  RayTrace.render(scene, trc=RayTrace.trcdepth)
end

"Render and vizualize scene"
showscene(scene) = imshow(rgbimg(RayTrace.render(scene; trc=RayTrace.trcdepth)))

## Test 
## ====
showscene(example_spheres(RayTrace.SimpleSphere))
showscene(example_spheres(NSphere))

function train_nraytrace(batch_size = 1)
  nrng = MersenneTwister(1234)
  refrng = MersenneTwister(1234)
  ref_tg() = trace(Any, () -> renderscene(RayTrace.SimpleSphere))
  n_tg() = trace(Any, () -> renderscene(NSphere))
  ref_tg, n_tg
end

## Distances
## =========
# δ(x::CInt, y::CInt) = mse(x.data, y.data)
# δ(x::CBool, y::CBool) = mse(x.data, y.data)

function optimizer(fs, consts)
  allparams = vcat(params.(fs)..., params.(consts)...)
  Flux.Optimise.ADAM(allparams)
end

"Train the networks"
function train()
  ref_tg, n_tg = train_nraytrace()
  nets = [rayintersect_net, NSphere_net]
  optimizer_ = optimizer(nets, [])
  step! = DSLearn.stepgen(ref_tg, n_tg, δ, optimizer_)
  RunTools.Optim.optimize(step!)
end

"Run Example"
function example()
  nscene = example_spheres(NSphere)
  RayTrace.render(nscene; trc=RayTrace.trcdepth)
end