"Data Structure Learning"
module DSLearn
using Spec
import Base.Iterators

include("util/misc.jl")
include("generator.jl")
include("trace.jl")
include("observe.jl")
include("train.jl")
include("optim/Optim.jl")

export observe!,
       trace,
       @grab

end