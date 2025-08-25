#!/usr/bin/env -S pixi run python

import os
import sys
import subprocess
from pathlib import Path


def submit():
    """Generate and submit an sbatch script on-the-fly"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate and submit SLURM sbatch scripts on-the-fly"
    )
    parser.add_argument("--job-name", "-J", required=True, help="Job name")
    parser.add_argument(
        "--partition", "-p", default="preempt", help="Partition to submit to"
    )
    parser.add_argument(
        "--qos", "-q", default=None, help="Quality of Service (QoS) to use"
    )
    parser.add_argument("--nodes", "-N", type=int, default=1, help="Number of nodes")
    parser.add_argument("--ntasks", "-n", type=int, default=1, help="Number of tasks")
    parser.add_argument(
        "--cpus-per-task", "-c", type=int, default=32, help="CPUs per task"
    )
    parser.add_argument("--memory", "-m", default="128G", help="Memory per node (e.g., 4G, 8000M)")
    parser.add_argument("--time", "-t", help="Time limit (e.g., 1:00:00)")
    parser.add_argument(
        "--gpus", "-g", default="1", help="Number of GPUs (e.g., 1, 2, or gpu:2)"
    )
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--error", "-e", help="Error file path")
    parser.add_argument("--email", help="Email address for notifications")
    parser.add_argument(
        "--email-type", help="Email notification type (BEGIN,END,FAIL,ALL)"
    )
    parser.add_argument(
        "--sbatch-opt", action="append", help="Additional sbatch options"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the generated script without submitting",
    )
    parser.add_argument(
        "--sync-logs",
        action="store_true",
        help="Sync logs after job completion",
    )

    args, command = parser.parse_known_args()

    if not command:
        parser.error("Command to execute is required")

    job_name = args.job_name
    partition = args.partition
    qos = args.qos
    nodes = args.nodes
    ntasks = args.ntasks
    cpus_per_task = args.cpus_per_task
    memory = args.memory
    time = args.time
    gpus = args.gpus
    output = args.output
    error = args.error
    email = args.email
    email_type = args.email_type
    additional_options = args.sbatch_opt
    dry_run = args.dry_run
    sync_logs = args.sync_logs

    # Add datetime suffix to job name
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name_with_timestamp = f"{job_name}_{timestamp}"

    # Create slurm_logs directory for this job
    logs_dir = Path("slurm_logs") / job_name_with_timestamp
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Generate sbatch script content
    script_content = "#!/bin/bash\n\n"

    # Add sbatch directives
    script_content += f"#SBATCH --job-name={job_name_with_timestamp}\n"
    script_content += f"#SBATCH --nodes={nodes}\n"
    script_content += f"#SBATCH --ntasks={ntasks}\n"
    script_content += f"#SBATCH --cpus-per-task={cpus_per_task}\n"

    if partition:
        script_content += f"#SBATCH --partition={partition}\n"

    if qos:
        script_content += f"#SBATCH --qos={qos}\n"

    if memory:
        script_content += f"#SBATCH --mem={memory}\n"

    if time:
        script_content += f"#SBATCH --time={time}\n"

    if gpus:
        script_content += f"#SBATCH --gres=gpu:{gpus}\n"

    # Set default output and error paths if not provided
    if not output:
        output = str(logs_dir / "stdout.txt")
    if not error:
        error = str(logs_dir / "stderr.txt")

    script_content += f"#SBATCH --output={output}\n"
    script_content += f"#SBATCH --error={error}\n"

    if email:
        script_content += f"#SBATCH --mail-user={email}\n"

    if email_type:
        script_content += f"#SBATCH --mail-type={email_type}\n"

    if additional_options:
        for opt in additional_options:
            script_content += f"#SBATCH {opt}\n"

    script_content += "\n"

    # Exit on any command failure
    script_content += "set -e\n"

    script_content += "set -x\n"
    script_content += "\n"

    # Set Hugging Face cache directory
    script_content += "# Set Hugging Face cache directory to use fast local storage\n"
    script_content += "export HF_HOME=/tmp/huggingface\n"
    script_content += "\n"

    # Activate pixi environment
    script_content += "# Activate pixi environment and run\n"
    script_content += "# Job execution\n"
    unescaped_command = ' '.join(command)
    # Escape square brackets with quotes
    escaped_command = unescaped_command.replace('[', '\'[').replace(']', ']\'')
    script_content += f"pixi run {escaped_command}\n"

    # Add data transfer back and cleanup if requested
    if sync_logs:
        script_content += "\n"
        script_content += "# Copy logs back to the object storage\n"
        script_content += f"gcloud storage rsync /tmp/lerobot-logs gs://cmu-gpucloud-$USER/lerobot-logs --recursive\n"

    if dry_run:
        print("Generated sbatch script:")
        print("=" * 50)
        print(script_content)
        print("=" * 50)
        return

    # Create sbatch file in the logs directory
    sbatch_file_path = logs_dir / f"{job_name_with_timestamp}.sbatch"
    with open(sbatch_file_path, "w") as sbatch_file:
        sbatch_file.write(script_content)

    try:
        # Submit the job
        result = subprocess.run(
            ["sbatch", str(sbatch_file_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"Job submitted successfully: {result.stdout.strip()}")
        print(f"Job files stored in: {logs_dir}")

    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e.stderr}", file=sys.stderr)
        exit(1)

    except FileNotFoundError:
        print(
            "Error: sbatch command not found. Make sure SLURM is installed and in your PATH.",
            file=sys.stderr,
        )
        exit(1)


if __name__ == "__main__":
    submit()
