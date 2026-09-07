"""Launch the external server unchanged, adding episode-boundary GC only.

The external agent owns its reset contract; do not duplicate its cache list.
This wrapper keeps the deployment reproducible without editing that worktree.
"""
import gc
import importlib.util
import os
from pathlib import Path
import sys


source = Path(sys.argv.pop(1)).resolve()
sys.path.insert(0, str(source.parent))
spec = importlib.util.spec_from_file_location("resident_memnav_backend", source)
server = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = server
spec.loader.exec_module(server)


def release_unused_memory():
    # The aggregator hook retains its last output independently of agent.reset.
    server.agent._dino_out[0] = None
    gc.collect()
    if server.torch.cuda.is_available():
        server.torch.cuda.synchronize()
        server.torch.cuda.empty_cache()
        server.torch.cuda.reset_peak_memory_stats()


@server.app.after_request
def collect_after_reset(response):
    if (server.request.path in {"/navigator_reset", "/navigator_reset_env"}
            and response.status_code == 200):
        release_unused_memory()
    return response


@server.app.route("/resident/release", methods=["POST"])
def release_episode():
    server.navigator_reset_env()
    release_unused_memory()
    return resident_status()


@server.app.route("/resident/status")
def resident_status():
    return server.jsonify({
        "resident_contract": "episode_reset_v1", "pid": os.getpid(),
        "memory_frames": server.agent.n,
        "depth_transactions": len(server.monocular_depth_transactions),
        "historical_depth_source": getattr(
            server.agent, "certified_reference_depth_source", "canonical"),
        "historical_depth_cached_frames": len(
            server.agent._certified_route_reference_depth_cache),
        "historical_depth_cache_bytes": sum(
            depth.nbytes + confidence.nbytes
            for depth, confidence in server.agent._certified_route_reference_depth_cache.values()),
        "buffer_root": server.agent.buffer_root,
        "episode_buffer": server.agent.rgb_dir,
        "cuda_allocated_bytes": server.torch.cuda.memory_allocated(),
        "cuda_reserved_bytes": server.torch.cuda.memory_reserved(),
    })


if __name__ == "__main__":
    print(f"Resident MemNav ready on :{server.args.port}", flush=True)
    server.app.run(host=server.args.host, port=server.args.port, threaded=False)
