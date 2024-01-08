import random
import json


TASKS = ["ioi", "docstring", "greaterthan", "tracr-reverse", "tracr-proportion", "induction"]

METRICS_FOR_TASK = {
    "ioi": ["kl_div", "logit_diff"],
    "tracr-reverse": ["l2"],
    "tracr-proportion": ["l2"],
    "induction": ["kl_div", "nll"],
    "docstring": ["kl_div", "docstring_metric"],
    "greaterthan": ["kl_div", "greaterthan"],
}

JOB = """
---

apiVersion: batch/v1

kind: Job
metadata:
  name: {NAME}
  labels:
    kueue.x-k8s.io/queue-name: farai
spec:
  ttlSecondsAfterFinished: 3600
  template:
    metadata:
      name: {NAME}
    spec:
      priorityClassName: low-batch
      containers:
      - name: dummy
        image: ghcr.io/arthurconmy/automatic-circuit-discovery:13057421
        command: {COMMAND}
        resources:
          # Request CPUs, limit memory and GPUs.
          requests:
            cpu: {CPU}
          limits:
            memory: "16G"
            nvidia.com/gpu: 0
        env:
        - name: WANDB_API_KEY
          valueFrom:
            secretKeyRef:
              key: api-key
              name: wandb
        - name: OMP_NUM_THREADS
          value: {CPU}
        volumeMounts:
        - name: parsed
          mountPath: {OUT_DIR}
      volumes:
      - name: parsed
        persistentVolumeClaim:
          claimName: acdc-parsed
      imagePullSecrets:  # Needed for private images
      - name: docker
      restartPolicy: Never
"""


def main(alg: str, task: str, CPU: str = "2"):
    seed = 1233778640
    random.seed(seed)

    OUT_DIR = "/parsed"

    commands = []
    for reset_network in [0, 1]:
        for zero_ablation in [0, 1]:
            for metric in METRICS_FOR_TASK[task]:
                if alg == "canonical" and (task == "induction" or metric == "kl_div"):
                    continue

                command = [
                    "python",
                    "notebooks/roc_plot_generator.py",
                    f"--task={task}",
                    f"--reset-network={reset_network}",
                    f"--metric={metric}",
                    f"--alg={alg}",
                    f"--device=cpu",
                    f"--torch-num-threads={CPU}",
                    f"--out-dir={OUT_DIR}",
                    f"--seed={random.randint(0, 2**31-1)}",
                ]
                if zero_ablation:
                    command.append("--zero-ablation")

                if (
                    alg == "acdc"
                    and task == "greaterthan"
                    and metric == "kl_div"
                    and not zero_ablation
                    and not reset_network
                ):
                    command.append("--ignore-missing-score")
                commands.append(command)

    for i, command in enumerate(commands):
        print(
            JOB.format(
                NAME=json.dumps(f"{alg}-{task}-{i:03d}"),
                COMMAND=json.dumps(command),
                CPU=json.dumps(CPU),
                OUT_DIR=json.dumps(OUT_DIR),
            )
        )


TASKS_FOR = {
    "acdc": TASKS,
    "16h": TASKS,
    "sp": TASKS,
    "canonical": TASKS,
}


if __name__ == "__main__":
    for alg in TASKS_FOR.keys():
        for task in TASKS_FOR[alg]:
            main(alg, task)
