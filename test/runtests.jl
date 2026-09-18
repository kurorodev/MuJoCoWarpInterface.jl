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
