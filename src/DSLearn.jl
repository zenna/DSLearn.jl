"Data Structure Learning"
module DSLearn
using Spec
import Base.Iterators

include("generator.jl")
include("trace.jl")

export observe!,
       trace

end