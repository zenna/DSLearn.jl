function parseatom(expr)

end

function parsefactors(expr)
end

""
function parsesummands(expr)
  num1 = parsefactors(expr)
  while true
    if op != "+" && op != "*"
      return num1
    elseif ok
    end
  end
end

"Evaluate the expression"
evalexpr(expr) = parsesummands(expr)