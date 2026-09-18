module MuJoCoWarpInterface

using PythonCall
using DLPack
using CUDA

include("backend.jl")
include("simulation.jl")
include("stepping.jl")

function __init__()
    _load_backend!()
end

export Simulation

export qpos
export qvel
export ctrl

export nq
export nv
export nu
export nworlds

export step!
export reset!

export backend_info

end