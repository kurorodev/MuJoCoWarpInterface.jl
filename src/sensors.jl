"""
    sensordata(sim)

Return the shared GPU sensor buffer with shape `(nsensordata(sim), nworlds(sim))`.
Reading this array does not trigger computation. Call `forward!` to recompute
readings for the current state, including after construction or `reset!`.
`step!` leaves the readings computed during the backend's integration step.
"""
sensordata(sim::Simulation) = sim.sensordata

"""Number of sensors defined in the model (a sensor may output multiple values)."""
nsensor(sim::Simulation) = pyconvert(Int, sim.mj_model.nsensor)

"""Total number of scalar sensor values per world."""
nsensordata(sim::Simulation) = size(sim.sensordata, 1)

"""
    sensor_names(sim)

Return sensor names in model order. Unnamed sensors are represented by `nothing`;
they remain accessible by their one-based index.
"""
function sensor_names(sim::Simulation)
    mj = _mujoco[]
    return Union{Nothing,String}[
        pyconvert(Union{Nothing,String}, mj.mj_id2name(
            sim.mj_model, mj.mjtObj.mjOBJ_SENSOR, i - 1,
        )) for i in 1:nsensor(sim)
    ]
end

"""
    sensor(sim, name::AbstractString)
    sensor(sim, index::Integer)

Return a zero-copy view of one sensor across all worlds, with shape
`(sensor_dimension, nworlds(sim))`, including for scalar sensors. Indices are
one-based and follow model order. Unknown names throw `ArgumentError`;
out-of-range indices throw `BoundsError`.

Cache the view outside simulation loops to avoid repeated model metadata lookups.
The view stays attached to the buffer across `step!`, `forward!`, and `reset!`.
"""
function sensor(sim::Simulation, index::Integer)
    1 <= index <= nsensor(sim) || throw(BoundsError(1:nsensor(sim), index))
    id = Int(index) - 1
    start = pyconvert(Int, sim.mj_model.sensor_adr[id]) + 1
    dim = pyconvert(Int, sim.mj_model.sensor_dim[id])
    return view(sensordata(sim), start:(start + dim - 1), :)
end

function sensor(sim::Simulation, name::AbstractString)
    mj = _mujoco[]
    id = pyconvert(Int, mj.mj_name2id(
        sim.mj_model, mj.mjtObj.mjOBJ_SENSOR, String(name),
    ))
    id >= 0 || throw(ArgumentError("Unknown sensor: $(repr(name))"))
    return sensor(sim, id + 1)
end
