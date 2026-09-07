# Ψ₀ with SONIC + Dex1-1 — Real-World Deployment

This follows Psi0's official four-process RTC flow. Dex1-1 is represented as a
dense virtual hand14 at the model interface and routed to two physical grippers
by the client adapter. Keep each command running in its own terminal.

## 1. Camera server on the robot

```bash
conda activate vision
cd ~/SONIC_psi0_release
python realsense_server.py
```

## 2. Psi0 policy server on the workstation

```bash
cd /path/to/Psi0
export CHECKPOINT_DIR=/path/to/psi0-sonic-run
export CHECKPOINT_STEP=40000
bash scripts/deploy/serve_psi0-rtc-sonic.sh
```

Wait for `Application startup complete` before continuing.

## 3. SONIC whole-body controller

```bash
cd /path/to/Psi0
bash real/scripts/deploy_psi0-sonic-rtc-robot.sh real
```

Confirm the real deployment. When `Init done.` appears, press **`]`** once and
wait for the robot to settle in its default standing reference. Do not enable
the ZMQ stream yet.

## 4. Psi0 client with Dex1-1 adapter

```bash
cd /path/to/Psi0
export SONIC_CONDA_PREFIX="$HOME/miniconda3/envs/sonic"
bash real/scripts/deploy_psi0-sonic-rtc-client.sh \
  --host 127.0.0.1 \
  --port 8014 \
  --camera-address tcp://192.168.123.164:5558 \
  --network enp4s0 \
  --instruction 'Pick up the object and place it in the container.' \
  --enable-dex1-live
```

Wait until the client prints `recv_action interval` and `Received action`.
Then press **Enter** once in the C++ controller terminal. The controller must
print `ZMQ STREAMING MODE: ENABLED` before policy execution begins.

To finish normally, press **Enter** again to return to the default reference,
stop the client with **Ctrl+C**, then press **O** in the C++ terminal to exit.
Press **O** immediately for an emergency stop.
