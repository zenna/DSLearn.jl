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
