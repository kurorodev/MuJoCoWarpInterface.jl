function step!(sim::Simulation)
    # Ensure writes performed by Julia/GPU backend are complete
    # before MJWarp reads the shared memory.
    CUDA.synchronize()

    _mjwarp[].step(
        sim.model,
        sim.data,
    )

    # Ensure MJWarp finished writing before Julia reads the state.
    _warp[].synchronize_device(
        sim.device,
    )

    return sim
end


function step!(
    sim::Simulation,
    actions::AbstractArray,
)
    size(actions) == size(sim.ctrl) ||
        throw(
            DimensionMismatch(
                "Expected actions with size $(size(sim.ctrl)), " *
                "got $(size(actions))"
            )
        )

    eltype(actions) == eltype(sim.ctrl) ||
        throw(
            ArgumentError(
                "Expected actions with eltype $(eltype(sim.ctrl)), " *
                "got $(eltype(actions))"
            )
        )

    # Avoid unnecessary copy if the user directly modified ctrl(sim).
    if actions !== sim.ctrl
        copyto!(sim.ctrl, actions)
    end

    return step!(sim)
end


function reset!(sim::Simulation)
    _mjwarp[].reset_data(
        sim.model,
        sim.data,
    )

    _warp[].synchronize_device(
        sim.device,
    )
    

    return sim
end