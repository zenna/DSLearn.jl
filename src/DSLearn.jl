"Data Structure Learning"
module DSLearn
using Spec
#using ZenUtils
import Base.Iterators

# include("util/misc.jl")
include("generator.jl")
include("trace.jl")
include("observe.jl")
include("train.jl")
"Return network associated with function `net(+. 1. 2)`"
function net end

export observe!,
       trace
end