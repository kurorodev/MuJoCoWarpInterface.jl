struct Simulation{Q,V,C}
    mj_model::Py
    model::Py
    data::Py

    qpos::Q
    qvel::V
    ctrl::C

    nworlds::Int
    device::String
end


function Simulation(
    model_path::AbstractString;
    nworlds::Integer = 1,
    device::Integer = 0,
    nconmax = nothing,
    njmax = nothing,
    nvmax = nothing,
)
    nworlds > 0 ||
        throw(ArgumentError("nworlds must be greater than zero"))

    isfile(model_path) ||
        throw(ArgumentError("MuJoCo model not found: $model_path"))

    CUDA.functional() ||
        error("CUDA is not functional on this system")

    CUDA.device!(device)

    device_name = "cuda:$device"

    mj  = _mujoco[]
    mjw = _mjwarp[]
    wp  = _warp[]

    # Initialize Warp and select GPU.
    wp.init()
    wp.set_device(device_name)

    # Host MuJoCo model.
    mj_model = mj.MjModel.from_xml_path(
        abspath(model_path),
    )

    # Device-side MJWarp model.
    model = mjw.put_model(mj_model)

    # Build keyword arguments dynamically.
    kwargs = Dict{Symbol,Any}(
        :nworld => Int(nworlds),
    )

    if nconmax !== nothing
        kwargs[:nconmax] = Int(nconmax)
    end

    if njmax !== nothing
        kwargs[:njmax] = Int(njmax)
    end

    if nvmax !== nothing
        kwargs[:nvmax] = Int(nvmax)
    end

    # Allocate N parallel worlds on GPU.
    data = mjw.make_data(
        mj_model;
        kwargs...,
    )

    mjw.reset_data(model, data)

    # Safe synchronization for v0.1.
    wp.synchronize_device(device_name)

    # ZERO-COPY:
    # these become CUDA.jl CuArrays backed by the
    # exact same allocations used by MJWarp.
    qpos_array = DLPack.from_dlpack(data.qpos)
    qvel_array = DLPack.from_dlpack(data.qvel)
    ctrl_array = DLPack.from_dlpack(data.ctrl)

    return Simulation(
        mj_model,
        model,
        data,
        qpos_array,
        qvel_array,
        ctrl_array,
        Int(nworlds),
        device_name,
    )
end

function Base.show(
    io::IO,
    sim::Simulation,
)
    print(
        io,
        "Simulation(",
        "nworlds=", sim.nworlds,
        ", nq=", nq(sim),
        ", nv=", nv(sim),
        ", nu=", nu(sim),
        ", device=\"", sim.device, "\"",
        ")",
    )
end

qpos(sim::Simulation) = sim.qpos
qvel(sim::Simulation) = sim.qvel
ctrl(sim::Simulation) = sim.ctrl

nworlds(sim::Simulation) = sim.nworlds

nq(sim::Simulation) = size(sim.qpos, 1)
nv(sim::Simulation) = size(sim.qvel, 1)
nu(sim::Simulation) = size(sim.ctrl, 1)