struct InfiniteSlices{A}
  data::A
  dim
  function InfiniteSlices{O}(f, slice::Integer) where {A}
    n, r = dimrem(size(A, dim), slice)
  end
end

"""
Generate infinite slices from an array

```jldoctest
A = rand(10, 10, 10)
it = InfiniteSlices(A, 1, 4)
```
"""
Base.eltype(::Type{InfiniteSlices{O}}) where {O} = A
Base.start(it::InfiniteSlices) = nothing
Base.next(it::InfiniteSlices, state) = (it.f(), nothing)
Base.done(it::InfiniteSlices, state) = false
Base.iteratorsize(::Type{<:InfiniteSlices}) = Base.IsInfinite()
Base.iteratoreltype(::Type{<:InfiniteSlices}) = Base.HasEltype()