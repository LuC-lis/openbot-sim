# OpenBot-Sim

**A from-scratch simulation and imitation-learning pipeline for the [OpenBot](https://www.openbot.org/) smartphone robot — built to study behaviour-cloning failure modes and sim-to-real transfer.**

This project reconstructs the robot's full perception–action loop in simulation, trains a vision-based driving policy by imitation, and uses controlled experiments to explain *why* such policies fail. It is an independent, self-directed project.

---

## Highlights

- **Simulation from scratch** (not a wrapper): differential-drive kinematics, a ground-plane ray-cast camera, road geometry, a task/reward, and a Gym-style loop — with the **observation/action interface matched exactly to the real OpenBot stack** (256×96 RGB image + discrete command → left/right wheel commands).
- **Full learning pipeline**: rule-based expert → synthetic dataset → TensorFlow CNN → TensorFlow Lite, plus a **Dataset Aggregation (DAgger)** loop with disturbance injection.
- **A diagnosed failure mode**: the default policy architecture quietly **ignores its camera input** and regresses to open-loop command following. Controlled ablations attribute the cause to **dropout (p = 0.5)**, with a sharp performance cliff between p = 0.4 and p = 0.5.
- **DAgger closes the gap**: closed-loop recovery success rises from **20% → 100%** over three iterations.
- **Engineered GPU workflow**: TensorFlow 2.15 + CUDA 12.2, reducing peak memory from **8.4 GB to 1.9 GB** (TFRecord streaming) for a **~4–5× speed-up**.

---

## Results

### 1. Dropout induces shortcut learning (sharp cliff)

Ablations on an identical dataset: only disabling dropout recovers the ability to use the image, while disabling batch-norm or the command concatenation does not. Scaling dropout from 0.0 to 0.5 shows learning up to p≈0.3, degradation at p≈0.4, and **total collapse at p=0.5**.

![dropout sweep](results/dropout_sweep.png)

### 2. DAgger improves closed-loop recovery

Starting from a policy trained only on expert demonstrations, each DAgger round lets the policy drive (with perturbations), records the expert's correction for the visited states, and retrains:

| Round | Frames | Hard-start success | Tracking error |
|------:|-------:|-------------------:|---------------:|
| 0 (behaviour cloning) | 1,600 | **20%** | 0.605 |
| 1 | 4,000 | **45%** | 0.405 |
| 2 | 6,165 | **95%** | 0.318 |
| 3 | 8,565 | **100%** | 0.314 |

![DAgger progress](results/dagger_progress.png)

### 3. Closed-loop driving (expert vs. learned policy)

![trajectory](results/m5_trajectory.png)

See also `results/drive_student.gif` (the learned policy) and `results/drive_teacher.gif` (the expert).

---

## Quick start

```bash
git clone <this repo> && cd openbot-sim
pip install -r requirements.txt

# --- pure-python demos (no TensorFlow needed) ---
python demos/m1_demo.py        # state + dynamics + world
python demos/m2_demo.py        # camera views        -> results/m2_views.png
python demos/m3_demo.py        # closed loop          -> results/m3_trajectory.png
python demos/m4_demo.py        # expert policy        -> results/m4_trajectory.png
python demos/tiny_nn.py        # a neural net written from scratch in numpy

# --- data → train → drive → evaluate ---
python scripts/export_data.py --train-episodes 4 --test-episodes 1   # generates output/dataset
python scripts/train.py --arch simple --epochs 15 --out models/student.tflite
python demos/m5_demo.py --model models/student.tflite
python scripts/evaluate.py --model models/student.tflite --episodes 20 --disturb 0.02
```

> If TensorFlow was installed from pip together with the `nvidia-*-cu12` wheels, run CUDA scripts through the helper:
> `./scripts/cuda_run.sh scripts/train.py --epochs 15`

---

## Repository layout

```
openbot_sim/     core simulation library (packaged, importable)
  state.py         robot state (x, y, theta)
  dynamics.py      differential-drive kinematics
  world.py         road geometry
  camera.py        forward pinhole camera + ground-plane ray casting
  task.py          reward and termination
  observation.py   observation assembly (the interface contract)
  spaces.py        observation/action spaces
  env.py           Gym-style reset()/step() loop
  policies.py      random / expert / TFLite policies
  nets.py          network definitions (self-contained)

scripts/         data, training, evaluation
  export_data.py   expert drives -> OpenBot-format dataset
  dagger.py        DAgger data aggregation
  train.py         configurable trainer (architecture / dropout)
  evaluate.py      multi-episode robustness evaluation
  check_model.py   verify a .tflite matches the interface
  cuda_run.sh      helper to expose pip CUDA libraries

experiments/     the diagnostic study
  probe_inmem.py     in-memory data control
  probe_lateral.py   image -> lateral-offset probing task
  probe_ablation.py  component ablations
  probe_dropout.py   dropout sweep

demos/           visual demos and teaching scripts
results/         figures and videos
models/          final trained policy (dg_s3.tflite, int8-quantized)
```

---

## The interface contract

The simulator reproduces OpenBot's real interface exactly, so trained `.tflite` models are drop-in compatible with the Android app:

| Quantity | Specification | Source in OpenBot |
|----------|---------------|-------------------|
| Image | `(96, 256, 3)` float32 in `[0, 1]` | `android/.../tflite/Autopilot.java`, `assets/config.json` |
| Command `cmd` | scalar in `{-1, 0, +1}` | `Autopilot.java`, `utils/Enums.java` |
| Action | `[left, right] ∈ [-1, 1]` | `vehicle/Control.java` |
| Serial encoding | `c<int(left*255)>,<int(right*255)>` | `python/infer.py` |

---

## The investigation, in one paragraph

A policy trained on an expert dataset appeared to work (low validation error) but drove poorly in closed loop. A systematic process — in-memory data controls (ruling out file/alignment bugs), a probing task showing the CNN *can* extract lateral offset from the image (MAE 0.024), component ablations, and a dropout sweep — isolated the cause: **heavy dropout (p = 0.5) makes the network minimise loss by predicting the command-conditional mean action, ignoring the image entirely (shortcut learning).** Lowering dropout restores vision-based steering, and a DAgger loop then raises closed-loop recovery from 20% to 100%. A detailed write-up (in Chinese) is in [`README.zh-CN.md`](README.zh-CN.md).

---

## Environment notes

- **CPU only**: `numpy`, `matplotlib`, `Pillow`, `tensorflow` — sufficient for the demos.
- **GPU (recommended for training)**: TensorFlow 2.15.1 requires **CUDA 12.2 / cuDNN 8.9**. If CUDA is provided via the pip `nvidia-*-cu12` wheels, TensorFlow will not find them automatically; `scripts/cuda_run.sh` sets `LD_LIBRARY_PATH` and `--xla_gpu_cuda_data_dir` accordingly. (Note: TF 2.15 uses CUDA 12.2, not 11.8 — a common source of confusion.)

---

## Attribution

The interface contract and network architecture follow [OpenBot](https://github.com/ob-f/OpenBot) (Müller & Koltun, *OpenBot: Turning Smartphones into Robots*, ICRA 2021, Apache-2.0). `openbot_sim/nets.py` is a minimal, self-contained re-implementation provided so this repository has no dependency on the OpenBot source tree.

## License

MIT — see [LICENSE](LICENSE).
