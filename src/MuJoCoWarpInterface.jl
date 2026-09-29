module MuJoCoWarpInterface

using PythonCall
using DLPack
using CUDA

include("backend.jl")
include("simulation.jl")
include("sensors.jl")
include("stepping.jl")

function __init__()
    _load_backend!()
end

export Simulation

export qpos
export qvel
export ctrl
export sensordata, sensor, sensor_names
export nsensor, nsensordata

export nq
export nv
export nu
export nworlds

export step!
export reset!
export forward!

export backend_info

end
