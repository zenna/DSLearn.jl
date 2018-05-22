"""
Train Data Structure
"""
function step(ref_ds, n_ds, δ, fs, consts)
  allparams = vcat(consts..., (collect(f[:parameters]()) for f in fs)...)
  adam = optim.Adam(allparams)
  function step!()
    ref_trace = ref_ds()
    net_trace = n_ds()
    losses = DSLearn.losses(net_trace, ref_trace, δ)
    loss = losses[1]
    loss[:backward]()
    adam[:step]()
    loss
  end
end

"Generate `step!` procedure to be called in `optimize``"
function stepgen(ref_ds,
                 n_ds,
                 δ,
                 optimizer;
                 accum = mean,
                 traceperstep=64)

  "Step function to be called in `optimize`"
  function step!(cb_data, callbacks)
    all_losses = []
    for i = 1:traceperstep

      # Run both reference and n data structure
      net_trace = n_ds()
      ref_trace = ref_ds()

      # Get all the losses
      @show losses = DSLearn.losses(net_trace, ref_trace, δ)
      for loss in losses
        push!(all_losses, loss)
      end
      # @show all_losses = vcat(all_losses, losses)
    end
    loss = accum(all_losses)
    # add_scalar!(writer, "Loss", data(loss), cb_data[:i])
    back!(loss)
    update!(optimizer)
    loss
    # @assert false
  end
end