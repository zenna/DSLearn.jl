"Is `T` an observable type?"
function isobservable end

"Observe `x`"
function observe!(nm::Symbol, x::T, trace_stack = global_trace_stack()) where T
  # @pre isobservable(x)
  OT = observe_type(T)
  trace_stack.trace[nm] = convert(OT, x)
  x
end

"Make a function observable"
macro observe(f)
  
end

function observe_type end