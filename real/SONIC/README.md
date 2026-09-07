# Ψ₀ with SONIC and Dex1-1

This adapter keeps SONIC's official launcher, PICO manager, data exporter, and
RTC client as the control flow. Psi0 only injects the installed Dex1-1 grippers:

- measured scalar gripper positions are projected to SONIC's hand14 interface;
- PICO trigger targets are sent to Dex1 and recorded as the same virtual hand14;
- policy hand14 output is mapped back to the physical grippers;
- the existing SIMPLE Remote Vision transport relays SONIC's `ego_view` to PICO.

The SONIC and SIMPLE submodules must be initialized. The wrapper uses an
existing conda `sonic` environment instead of creating additional environments:

```bash
git submodule update --init --recursive
export SONIC_CONDA_PREFIX="$HOME/miniconda3/envs/sonic"
```

`SONIC_DIR` may point to a separate official SONIC checkout. Otherwise it
defaults to `third_party/GR00T-WholeBodyControl`.

## Camera server

Run SONIC's official composed camera server on the robot:

```bash
python -m gear_sonic.camera.composed_camera \
  --ego-view-camera realsense \
  --port 5555
```

## Data collection

The entry point delegates tmux creation and process lifecycle to SONIC's
official `launch_data_collection.py`. It replaces only the manager, exporter,
and viewer commands with their Dex1-aware counterparts:

```bash
bash real/SONIC/scripts/collect_psi0-sonic-data.sh \
  --camera-host 192.168.123.164 \
  --network enp4s0 \
  --task-prompt 'Pick up the object and place it in the container.' \
  --dataset-name sonic_dex1 \
  --root-output-dir "$HOME/data"
```

The official camera-viewer pane is used for the PICO ego-view relay. In Remote
Vision, select H.264 and connect to `<workstation-ip>:13579`. Pass
`--no-camera-viewer` to disable this relay.

The manager retains SONIC's official mode state machine. Because Remote Vision
uses physical B and the four-face-button chord, the wrapper makes only these
input substitutions:

- hold Right Stick Click for 0.6 seconds: policy on/off;
- `Left Grip + A`: start/stop an episode;
- `Left Grip + Y`: discard an episode;
- `A + X`: switch between Planner and full-body POSE.

Press `Ctrl+\` to terminate the official tmux session. No extra episode-quality
gate is added; saving and discarding use SONIC's exporter behavior.

## Conversion

The official raw SONIC schema remains unchanged. Dex1 recordings add only an
`end_effector=dex1_virtual14` marker. Conversion preserves the dense hand14
targets from SONIC's `teleop.left_hand_joints` and
`teleop.right_hand_joints` columns:

```bash
"$SONIC_CONDA_PREFIX/bin/python" scripts/data/raw_sonic_to_psi_lerobot.py \
  --data-root "$HOME/data/sonic_dex1" \
  --work-dir "$PWD/data/sonic/lerobot" \
  --repo-id sonic_dex1 \
  --robot-type g1 \
  --end-effector dex1_virtual14

"$SONIC_CONDA_PREFIX/bin/python" scripts/data/calc_modality_stats.py \
  --task-dir "$PWD/data/sonic/lerobot/sonic_dex1"
cp data/sonic/lerobot/sonic_dex1/meta/stats.json \
   data/sonic/lerobot/sonic_dex1/meta/stats_psi0.json
```

Training then uses Psi0's existing official SONIC configuration without model
or trainer changes.

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md). Deployment keeps the official four-process
flow and wraps only the RTC client so Dex1 state and action use the same mapping
as collection and conversion.
