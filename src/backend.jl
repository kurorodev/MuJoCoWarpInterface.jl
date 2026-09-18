const _mujoco = Ref{Py}()
const _mjwarp = Ref{Py}()
const _warp   = Ref{Py}()

function _load_backend!()
    _mujoco[] = pyimport("mujoco")
    _mjwarp[] = pyimport("mujoco_warp")
    _warp[]   = pyimport("warp")

    return nothing
end

function backend_info()
    metadata = pyimport("importlib.metadata")

    return (
        mujoco = pyconvert(
            String,
            metadata.version("mujoco"),
        ),

        mujoco_warp = pyconvert(
            String,
            metadata.version("mujoco-warp"),
        ),

        warp = pyconvert(
            String,
            metadata.version("warp-lang"),
        ),
    )
end