import argparse
import os
from pathlib import Path

from unlearning_bench.paths import CONFIG_DIR

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", required=True)
    p.add_argument("--jobs-per-gpu", type=int, default=1)
    p.add_argument("--output", default="launch_jobs.sh")
    p.add_argument("--filters", default=None)
    args = p.parse_args()

    gpus = [int(x) for x in args.gpus.split(",") if x]
    filters = [x for x in (args.filters or "").split(",") if x]

    # gather configs per GPU
    gpu_jobs = {g: [] for g in gpus}
    queue = sorted(
        f for f in CONFIG_DIR.iterdir()
        if f.suffix == ".yml"
        and (not filters or all(filt in f.name for filt in filters))
    )
    for i, cfg in enumerate(queue):
        gpu = gpus[i % len(gpus)]
        gpu_jobs[gpu].append(cfg.name)

    # determine log filename
    script_path = Path(args.output)
    base = script_path.stem
    log_file = f"pdb_{base}.txt"

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f'LOG_FILE="{log_file}"',
        ': > "$LOG_FILE"',
        ""
    ]

    for g in gpus:
        jobs = gpu_jobs[g]
        if not jobs:
            continue

        lines.append("(")
        for idx, cfg in enumerate(jobs, start=1):
            cmd_str = f'CUDA_VISIBLE_DEVICES={g} python run.py --c {cfg}'

            # inline the pipeline; if "(Pdb)" appears, append the full cmd_str
            lines.append(
                f'{cmd_str} 2>&1 | '
                f'tee >(grep -q "(Pdb)" && echo "{cmd_str}" >> "$LOG_FILE") &'
            )

            if idx % args.jobs_per_gpu == 0 or idx == len(jobs):
                lines.append("wait")
        lines.append(") &")

    lines.append("wait")

    with open(args.output, "w") as f:
        f.write("\n".join(lines))
    os.chmod(args.output, 0o755)

if __name__ == "__main__":
    main()

