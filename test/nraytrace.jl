using Flux
import RayTrace: Geometry, Scene, Vec3, Ray, Intersection

"Learned Geometry"
abstract type NGeometry <: Geometry

"Learned sphere sphere"
struct NSphere{T} <: NGeometry
  data::T
end
size(::Type{NSphere}) = (4,) #TODO: parameterize this
size(::Type{Vec3}) = (3,)

"Continuous Ray tyoe"
struct CRay{T}
  data::T
end

"Continuous Intersection type"
struct CIntersect{T}
  data::T
end

## Conversions
## ===========
DSLearn.observe_type(::Type{<:NInt}) = CInt
DSLearn.observe_type(::Type{Int}) = CInt
DSLearn.observe_type(::Type{<:NBool}) = CBool
DSLearn.observe_type(::Type{Bool}) = CBool

Base.convert(::Type{CRay}, ::Ray) = CRay(vcat(r.orig, r.dir))
Base.convert(::Type{})

Base.size(RayTrace.ray) = (6,)

NSphere_net = mlp((Vec3, (1,)), (NSphere,))
NSphere(xyz::Vec3, r) = NSphere_net(xyz, r)

rayintersect_net = mlp((Ray, NSphere), (Intersection))
rayintersect(r::Ray, s::NSphere)::Intersection = rayintersect_net(r, s)

## Test
## ====
"Some example spheres which should create actual image"
function example_spheres(SphereT)
  RayTrace.ListScene(
   [SphereT(Vec3([0.0, -10004, -20]), 10000.0)
    SphereT(Vec3([0.0,      0, -20]),     4.0),
    SphereT(Vec3([5.0,     -1, -15]),     2.0),
    SphereT(Vec3([5.0,      0, -25]),     3.0),
    SphereT(Vec3([-5.5,      0, -15]),    3.0),
    # light (emission > 0)
    SphereT(Vec3([0.0,     20.0, -30]),  3.0)]
end

function renderscene(SphereT::DataType)
  scene = example_spheres(SphereT)
  RayTrace.render(scene)
end

function train_arithmetic(batch_size = 1)
  nrng = MersenneTwister(1234)
  refrng = MersenneTwister(1234)
  ref_tg() = trace(Any, () -> renderscene(refrng, SimpleSphere))
  n_tg() = trace(Any, () -> ex1(nrng, NSphere))
  ref_tg, n_tg
end