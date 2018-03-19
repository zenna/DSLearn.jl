# Split the batch size
import RayTrace: Geometry, Scene

"Learned Geometry"
abstract type NGeometry <: Geometry

"A neural sphere"
struct NSphere{T} <: NGeometry
  data::T
end

struct NScene <: Scene
  data::T
end

function Base.isnull(s::Scene)
end

"Add geometric object to a scene"
function Base.push!(nscene::NScene, geom::NGeometry)
  ## Neural Network here
end

struct NBool{T}
  data::T
end

function Base.convert(Bool, NBool)
end

function doesintersect(r::Ray, s::NSphere)::Nbool
end

"Normal of ray intersecting with sphere"
function normal(r::Ray, s::Sphere)
end

function tansparency(s::Sphere)
end

function reflection(s::Sphere)
end