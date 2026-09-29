using Test
using CUDA
using PythonCall
using MuJoCoWarpInterface

@testset "MuJoCoWarpInterface" begin
    model_path = joinpath(
        @__DIR__,
        "models",
        "pendulum.xml",
    )
    sensor_model_path = joinpath(@__DIR__, "models", "sensors.xml")

    @testset "Sensor lookup without GPU" begin
        mj = pyimport("mujoco")
        model = mj.MjModel.from_xml_path(sensor_model_path)
        readings = reshape(Float32.(1:24), 8, 3)
        # Exercise metadata and indexing using a host buffer, without CUDA.
        sim = Simulation(model, pybuiltins.None, pybuiltins.None,
                         zeros(Float32, 1, 3), zeros(Float32, 1, 3),
                         zeros(Float32, 1, 3), readings, 3, "cpu")
        @test nsensor(sim) == 4
        @test nsensordata(sim) == 8
        @test sensor_names(sim) == ["angle", "velocity", "tip_position", nothing]
        @test sensordata(sim) === readings
        @test size(sensor(sim, "angle")) == (1, 3)
        @test sensor(sim, "tip_position") == readings[3:5, :]
        @test sensor(sim, 4) == readings[6:8, :]
        @test parent(sensor(sim, 3)) === readings
        @test_throws ArgumentError sensor(sim, "missing")
        @test_throws ArgumentError sensor(sim, "")
        @test_throws BoundsError sensor(sim, 0)
        @test_throws BoundsError sensor(sim, 5)

        empty_model = mj.MjModel.from_xml_path(model_path)
        empty_sim = Simulation(empty_model, pybuiltins.None, pybuiltins.None,
                               zeros(Float32, 1, 3), zeros(Float32, 1, 3),
                               zeros(Float32, 1, 3), zeros(Float32, 0, 3), 3, "cpu")
        @test nsensor(empty_sim) == nsensordata(empty_sim) == 0
        @test isempty(sensor_names(empty_sim))
        @test size(sensordata(empty_sim)) == (0, 3)
        @test_throws BoundsError sensor(empty_sim, 1)
        @test_throws ArgumentError sensor(empty_sim, "angle")
    end

    # These tests run on every CI machine, including GitHub-hosted
    # runners that do not provide an NVIDIA GPU.
    @testset "Package and backend" begin
        info = backend_info()

        @test !isempty(info.mujoco)
        @test !isempty(info.mujoco_warp)
        @test !isempty(info.warp)

        @test isfile(model_path)
    end

    # Standard GitHub-hosted runners do not have an NVIDIA GPU.
    # GPU-specific tests are therefore only executed when CUDA is usable.
    if CUDA.functional()
        @info "CUDA available; running MJWarp GPU tests"

        @testset "Simulation construction" begin
            sim = Simulation(
                model_path;
                nworlds = 1024,
            )

            @test nworlds(sim) == 1024

            @test nq(sim) == 1
            @test nv(sim) == 1
            @test nu(sim) == 1

            @test size(qpos(sim)) == (1, 1024)
            @test size(qvel(sim)) == (1, 1024)
            @test size(ctrl(sim)) == (1, 1024)

            @test qpos(sim) isa CuArray
            @test qvel(sim) isa CuArray
            @test ctrl(sim) isa CuArray
            @test sensordata(sim) isa CuArray
            @test size(sensordata(sim)) == (0, 1024)
            @test nsensor(sim) == nsensordata(sim) == 0
        end

        @testset "Zero-copy interoperability" begin
            sim = Simulation(
                model_path;
                nworlds = 1024,
            )

            warp_qpos_ptr =
                pyconvert(UInt, sim.data.qpos.ptr)

            warp_qvel_ptr =
                pyconvert(UInt, sim.data.qvel.ptr)

            warp_ctrl_ptr =
                pyconvert(UInt, sim.data.ctrl.ptr)

            @test UInt(pointer(qpos(sim))) ==
                  warp_qpos_ptr

            @test UInt(pointer(qvel(sim))) ==
                  warp_qvel_ptr

            @test UInt(pointer(ctrl(sim))) ==
                  warp_ctrl_ptr
        end

        @testset "Sensors on GPU" begin
            sim = Simulation(sensor_model_path; nworlds = 3)
            readings = sensordata(sim)
            angle = sensor(sim, "angle")
            tip = sensor(sim, "tip_position")
            @test readings isa CuArray
            @test size(readings) == (8, 3)
            @test UInt(pointer(readings)) == pyconvert(UInt, sim.data.sensordata.ptr)
            @test parent(tip) === readings

            positions = reshape(Float32[0.2, 0.4, 0.6], 1, 3)
            velocities = reshape(Float32[0.1, 0.2, 0.3], 1, 3)
            copyto!(qpos(sim), positions)
            copyto!(qvel(sim), velocities)
            @test forward!(sim) === sim
            @test Array(angle) ≈ positions
            @test Array(sensor(sim, "velocity")) ≈ velocities
            @test Array(tip) ≈ vcat(cos.(positions), sin.(positions), zeros(Float32, 1, 3))
            @test Array(sensor(sim, 4)) ≈ vcat(zeros(Float32, 2, 3), velocities)

            # Compare step-time readings with the CPU MuJoCo pipeline, which
            # computes sensors during the step, before Euler integration.
            mj = pyimport("mujoco")
            host_data = mj.MjData(sim.mj_model)
            host_data.qpos[0] = positions[1]
            host_data.qvel[0] = velocities[1]
            mj.mj_step(sim.mj_model, host_data)
            step!(sim)
            expected = pyconvert(Vector{Float64}, host_data.sensordata)
            @test Array(readings)[:, 1] ≈ expected rtol = 1e-4 atol = 1e-6

            @test forward!(sim) === sim
            @test Array(angle) ≈ Array(qpos(sim))
            @test sensordata(sim) === readings
            reset!(sim)
            forward!(sim)
            @test Array(angle) == zeros(Float32, 1, 3)
            @test Array(tip) ≈ repeat(Float32[1, 0, 0], 1, 3)
            @test sensordata(sim) === readings
        end

        @testset "Physics stepping" begin
            sim = Simulation(
                model_path;
                nworlds = 1024,
            )

            reset!(sim)

            actions = CUDA.fill(
                1.0f0,
                size(ctrl(sim)),
            )

            for _ in 1:100
                step!(sim, actions)
            end

            result = Array(
                qpos(sim)[:, 1:4]
            )

            @test all(result .> 0.0f0)

            @test all(
                isapprox.(
                    result,
                    0.0584541f0;
                    atol = 1f-5,
                )
            )
        end

        @testset "Reset" begin
            sim = Simulation(
                model_path;
                nworlds = 1024,
            )

            ctrl(sim) .= 1.0f0

            for _ in 1:10
                step!(sim)
            end

            @test any(
                Array(qpos(sim)) .!= 0.0f0
            )

            reset!(sim)

            @test all(
                Array(qpos(sim)) .== 0.0f0
            )

            @test all(
                Array(qvel(sim)) .== 0.0f0
            )
        end

    else
        @info "CUDA unavailable; skipping MJWarp GPU tests"
        @test true
    end
end
