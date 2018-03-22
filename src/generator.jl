"Infinite generator of batches from data"
function infinite_batches(data, batch_dim, batch_size, nelems = size(data, batch_dim))
  ids = Iterators.partition(Iterators.cycle(1:nelems), batch_size)
  (slicedim(data, batch_dim, id) for id in ids)
end