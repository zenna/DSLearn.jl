using DSLearn
using Spec

walktests(DSLearn)
walktests(DSLearn; test_dir = joinpath(Pkg.dir(string(DSLearn)), "test", "models"))
