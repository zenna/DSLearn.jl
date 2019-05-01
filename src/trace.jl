
"Trace of observable values from executing a program"
struct Trace{T}
  trace::Dict{Symbol, T}
end

global global_trace_stack_ = Trace(Dict{Symbol, Any}())

function global_trace_stack()
  global global_trace_stack_
  return global_trace_stack_
end

"Set global trace to `trace_stack`"
function with_trace(f, trace_stack::Trace)
  global global_trace_stack_ = trace_stack
  f()
  global_trace_stack = Trace(Dict{Symbol, Any}())
end

"Execute function `f(args)` and record trace of observable values"
function trace(trace_stack::Trace, f::Function, args...; kwargs...)::Trace
  with_trace(trace_stack) do
    f(args...; kwargs...)
  end
  trace_stack
end

function trace(T::Type, f::Function, args...; kwargs...)::Trace
  trace(Trace(Dict{Symbol, T}()), f, args...; kwargs...)
end

trace(f::Function, args...; kwargs...) = trace(Any, args...; kwargs...)

"Losses between corresponding elements of trace"
function losses(t1::Trace, t2::Trace, δ)
  [δ(t1.trace[nm], t2.trace[nm]) for (nm, val) in t1.trace if nm in keys(t2.trace)]
end