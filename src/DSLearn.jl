"Data Structure Learning"
module DSLearn
using Spec

"Trace of observable values from executing a program"
struct Trace
  trace::Vector
end

global global_trace_stack_ = Trace([])

function global_trace_stack()
  global global_trace_stack_
  return global_trace_stack_
end

"Set global trace to `trace_stack`"
function with_trace(f, trace_stack::Trace)
  global global_trace_stack_ = trace_stack
  f()
  global global_trace_stack = Trace([])
end

"Is `T` an observable type?"
function isobservable(T::DataType) end

"Observe `x`"
function observe!(x)
  @pre isobservable(x)
  trace_stack = global_trace_stack()
  push!(trace_stack.trace, x)
  x
end

"Execute function `f(args)` and record trace of observable values"
function Base.trace(f::Function, args...; kwargs...)
  trace_stack = Trace([])
  with_trace(trace_stack) do
    f(args...; kwargs...)
  end
end

export observe!,
       trace

end