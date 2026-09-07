"""Resident model lifecycle helpers. Call only under the GPU lifecycle flock.

Parking follows Jetson shutdown: remove the hub, drain NavDP (which can call
MemNav), then reset MemNav. No recorder, dataset or RGB evidence is deleted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from urllib.request import Request, urlopen


def tmux(*args, check=True):
    return subprocess.run(["tmux", *args], check=check, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def environment(session, name):
    result = tmux("show-environment", "-t", session, name, check=False)
    return result.stdout.strip().removeprefix(name + "=")


def signature(config):
    gpu = config["sites"]["gpu"]
    repo = Path(gpu["repository"])
    models = gpu["models"]
    files = set((repo / "baselines/navdp").glob("*.py"))
    files.update((repo / "deployment/gpu/scripts").glob("*.sh"))
    files.update((repo / "deployment/gpu").glob("*.py"))
    external = Path(models["memnav_source_root"])
    for root in [external / "NavDP/baselines/memnav",
                 external / "MemNavData", Path(models["lingbot_repository"])]:
        files.update(root.rglob("*.py"))
    digest = hashlib.sha256()
    digest.update(json.dumps({
        "gpu": gpu, "eager_depth_cache": config["cec"]["eager_depth_cache"],
        "historical_depth_source": config["cec"].get(
            "historical_depth_source", "canonical"),
        "revision": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
    }, sort_keys=True).encode())
    for path in sorted(files):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    # Do not reread multi-GB immutable weights at every phase boundary.
    # A changed path/size/mtime/ctime forces a cold start, as does changed code.
    for key in ("memnav_checkpoint", "navdp_checkpoint", "lingbot_weights"):
        stat = Path(models[key]).stat()
        digest.update(str((key, stat.st_size, stat.st_mtime_ns,
                           stat.st_ctime_ns)).encode())
    return digest.hexdigest()


def service(gpu, name, release=False):
    url = f"http://127.0.0.1:{gpu['ports'][name]}/resident/"
    request = Request(url + ("release" if release else "status"),
                      data=b"{}" if release else None,
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=120 if release else 5) as response:
        result = json.load(response)
    if result.get("resident_contract") != "episode_reset_v1":
        raise RuntimeError(f"{name}: incompatible resident service")
    if (result.get("memory_frames", 0) != 0
            or any(result.get("queue_lengths", []))
            or result.get("depth_transactions") != 0):
        raise RuntimeError(f"{name}: episode state was not cleared: {result}")
    return result


def park(config):
    gpu = config["sites"]["gpu"]
    session = gpu["session"]
    if tmux("has-session", "-t", session, check=False).returncode:
        print("GPU models already stopped")
        return
    if environment(session, "MEMNAV_RESIDENT_MANAGER") != "v1":
        raise RuntimeError("Refusing to park an unmanaged/pre-existing GPU session")
    if environment(session, "MEMNAV_RESIDENT_STATE") == "parked":
        print("GPU models already parked")
        return
    if environment(session, "MEMNAV_CONFIG_ID") != config["config_id"]:
        raise RuntimeError("Refusing to park another active GPU configuration")
    tmux("set-environment", "-t", session, "MEMNAV_RESIDENT_STATE", "parking")
    tmux("kill-window", "-t", session + ":hub", check=False)
    started = time.monotonic()
    try:
        receipts = {name: service(gpu, name, release=True)
                    for name in ("navdp", "memnav")}
        tmux("set-environment", "-t", session, "MEMNAV_RESIDENT_STATE", "parked")
        result = {"state": "parked", "config_id": config["config_id"],
                  "unix_time": time.time(), "elapsed_s": time.monotonic() - started,
                  "services": receipts}
        log = Path(gpu["runtime_root"]) / "logs/resident_lifecycle.jsonl"
        with log.open("a") as stream:
            stream.write(json.dumps(result, sort_keys=True) + "\n")
        print(json.dumps(result, sort_keys=True))
    except Exception:
        # A failed reset is never eligible for reuse. Cold start next time.
        tmux("kill-session", "-t", session, check=False)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["signature", "park", "verify-idle"])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.action == "signature":
        print(signature(config))
    elif args.action == "park":
        park(config)
    else:
        print(json.dumps({name: service(config["sites"]["gpu"], name)
                          for name in ("navdp", "memnav")}))
