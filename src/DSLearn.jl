"Data Structure Learning"
module DSLearn
using Spec
import Base.Iterators

include("generator.jl")
include("trace.jl")
include("train.jl")
include("optim/Optim.jl")

export observe!,
       trace

end