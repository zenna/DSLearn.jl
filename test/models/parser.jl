using DSLearn

struct NStack{T}
  data::T
end
Base.size(::Type{NStack}) = (10,)

struct NScratch{T}
  data::T
end
Base.size(::Type{NScratch}) = (10,)
"Construct empty scratch pad"
NScratch() = 3

struct NSymbol
  data::T
end
Base.size(::Type{NScratch}) = (20,)

struct NString
  data::T
end

"Controls whether stack is pushed to or popped from"
g(x::NSymbol, ::NScratch, ::NBool) = net(g, NScratch, NBool)

"Does parsing logic"
parselogic(x::NSymbol, ::NScratch, ::NBool) = net(g, NScratch, NBool)

"Parse a sequence of strings into an expression"
function parse(expr)
  stack = NStack()
  scratch = NScratch()
  for symbol in expr
    dopush, scratch = g(symbol, scratch, isempty(stack))
    if dopush
      push!(stack, symbol)
    else
      val = pop!(stack)
      if g(ina[i], scratch, vdal)
        da
      end
    end
  end
end

## Training
