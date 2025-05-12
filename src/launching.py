import argparse, os
from paths import CONFIG_DIR

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", required=True)
    p.add_argument("--jobs-per-gpu", type=int, default=1)
    p.add_argument("--output", default="launch_jobs.sh")
    p.add_argument("--method", default=None)
    a = p.parse_args()

    gpus = [int(x) for x in a.gpus.split(",") if x]
    gpu_jobs = {g: [] for g in gpus}
    queue = sorted([f for f in CONFIG_DIR.iterdir() if (f.suffix == ".yml" and (a.method is None or a.method in f.name))])
    for i, cfg in enumerate(queue):
        gpu_jobs[gpus[i % len(gpus)]].append(cfg.name)

    lines = ["#!/usr/bin/env bash", "set -e"]
    for g in gpus:
        jobs = gpu_jobs[g]
        if not jobs:
            continue
        lines.append("(")
        for i, cfg in enumerate(jobs, 1):
            lines.append(f"CUDA_VISIBLE_DEVICES={g} python run.py --c {cfg} &")
            if i % a.jobs_per_gpu == 0 or i == len(jobs):
                lines.append("wait")
        lines.append(") &")
    lines.append("wait\n")

    with open(a.output, "w") as f:
        f.write("\n".join(lines))
    os.chmod(a.output, 0o755)

if __name__ == "__main__":
    main()

