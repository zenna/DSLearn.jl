using Flux
import RayTrace: Geometry, Scene, Vec3

"Learned Geometry"
abstract type NGeometry <: Geometry

"Learned sphere sphere"
struct NSphere{T} <: NGeometry
  latent::T
end
size(::Type{NSphere}) = (4,) #TODO: parameterize this
size(::Type{Vec3}) = (3,)

NSphere_net = mlp((Vec3, NSphere), (NSphere,))
NSphere(xyz::Vec3, r) = NSphere_net(xyz, r)

rayintersect_net = mlp((Ray, NSphere), (Intersection))
rayintersect(r::Ray, s::NSphere)::Intersection = rayintersect_net(r, s)

## Conversions
## ===========

## Test
## ====
"Some example spheres which should create actual image"
function example_spheres()
  RayTrace.ListScene(
   [NSphere(Vec3([0.0, -10004, -20]), 10000.0)
    NSphere(Vec3([0.0,      0, -20]),     4.0),
    NSphere(Vec3([5.0,     -1, -15]),     2.0),
    NSphere(Vec3([5.0,      0, -25]),     3.0),
    NSphere(Vec3([-5.5,      0, -15]),    3.0),
    # light (emission > 0)
    NSphere(Vec3([0.0,     20.0, -30]),  3.0)]
end

function show_img()
  spheres = example_spheres()
  img_ = RayTrace.render(scene)
  img = rgbimg(img_)
  ImageView.imshow(img)
end