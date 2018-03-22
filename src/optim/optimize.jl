import IterTools

@enum Stage Pre Run Post

apl(f, data) = f(data)
take!(x::Real) = x
take!(x::Array{<:Real}) = x
take!(f::Function) = f()
take1(rep) = collect(Base.Iterators.take(rep, 1))[1]
function take1(imap::IterTools.IMap)
  for i in imap
    return i
  end
end

"""
Optimization.
# Arguments
- `step!(cb_data, callbacks)`: takes a gradient step and returns the loss
   must also call populate cb_data with any additiona data and call callbacks
   e.g..,
   ```cb_data_ = merge(cb_data, @NT(loss = cur_loss))
      foreach(cb->cb(cb_data_), callbacks)```
- `writer`: Summary writer to log results to tensorboardX
- `close_writer`: Close writer after finish?
- `pre_callbacks`: functions/generators called before optimization
- `callbacks`: functions called with data every iteration, e.g for viz
- `post_callbacks`: functions/generators called after optimization
- `maxiters`: num of iterations
- `cont`: function to determine when to stop (overrides maxiters)
- `resetlog`: reset log data after every iteration if true
- `logdir`: directory to store data/logs (used by callbacks)
- `optimize`: optimize? (compute grads/change weights)
- `start_i`: what index is this starting at (used by callbacks)
"""
function optimize(step!::Function;
                  pre_callbacks = [],
                  callbacks = [],
                  post_callbacks = [],
                  cont = data -> data[:i] < 100000,
                  resetlog::Bool=true,
                  logdir::String="",
                  optimize::Bool=true,
                  start_i::Integer=0)
  i = 0
  cb_data = Dict{Symbol, Any}(:start_i => start_i, :i => i, :Stage => Pre)
  
  # Called once before optimization
  foreach(cb->apl(cb, cb_data), pre_callbacks)
  lens(:pre_opt, start_i = start_i, i = i)

  while cont(cb_data)
    cb_data = Dict{Symbol, Any}(:start_i=>start_i, :i=>i, :Stage=>Run)
    if optimize
      step!(cb_data, callbacks)
    end
    # foreach(cb->apl(cb, cb_data), callbacks)
    i += 1
    # resetlog && reset_log()
  end

  # Post Callbacks
  lens(:post_opt, start_i = start_i, i = i)
  cb_data = Dict{Symbol, Any}(:start_i=>start_i, :i=>i, :Stage=>Post)
  foreach(cb->apl(cb, cb_data), post_callbacks)
end