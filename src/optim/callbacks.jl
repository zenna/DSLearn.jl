using JLD2, FileIO

"Record standard run metrics to dataframe"
function recordrungen(runname::Symbol)
  df = DataFrame(runname = Symbol[],
                 iteration = Int[],
                 loss = Float64[],
                 systime = Float64[])
                #  input = Vector{Float64}[],
                #  output = Vector{Float64}[])
  i = 0
  function recordrun(cbdata)
    row = [runname, i, cbdata.loss, time()]
    # , [cbdata.input...], [cbdata.output...]]
    push!(df, row)
    i = i + 1
  end
  df, recordrun
end

"Save dataframe to file"
function savedfgen(path::String, df::DataFrame)
  cbdata -> savedf(path, df)
end

"Save dataframe to file"
function savedfgen(rd::Dict, df::DataFrame)
  path = joinpath(rd[:logdir], "$(rd[:runname]).jld2")
  savedfgen(path, df)
end

"Higher order function that makes a callback run just once every n"
function everyn(callback, n::Integer)
  function everyncb(data)
    if data.i % n == 0
      callback(data)
    end
  end
  return everyncb
end

function printloss(data)
  println("loss: ", data.loss)
end


# "Has the optimization converged?"
# function converged(every, print_change=True, change_thres=-0.000005):
#   function converged_gen(every)
#     running_loss = 0.0
#     last_running_loss = 0.0
#     show_change = False
#     cont = True
#     while true:
#       data = yield cont
#       if data.loss is None:
#         continue
#       running_loss += data.loss
#       if (data.i + 1) % every == 0:
#         if show_change:
#           change = (running_loss - last_running_loss)
#           print('absolute change (avg over {}) {}'.format(every, change))
#           if last_running_loss != 0:
#             relchange = change / last_running_loss
#             per_iter = relchange / every
#             print('relative_change: {}, per iteration: {}'.format(relchange,
#                                                                   per_iter))
#             if per_iter > change_thres:
#               print("Relative change insufficeint, stopping!")
#               cont = False
#         else:
#           show_change = True
#         last_running_loss = running_loss
#         running_loss = 0.0

#   gen = converged_gen(every)
#   next(gen)
#   return gen