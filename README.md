# Double Inverted Pendulum with Pymunk and PPO

This project builds a **double inverted pendulum simulation** using `pymunk` and `pygame`, and trains an agent using **PPO (Proximal Policy Optimization)** from `stable-baselines3`. Everything is designed to be reproducible using Docker.

---

## Screenshots

You can add your own screenshots here after running the project:

* Reward comparison graph:
  `reward_comparison.png`

* Initial agent behavior (early training):
  `media/agent_initial.gif`

* Final trained agent behavior:
  `media/agent_final.gif`

---

## Demo Video

If you record a demo (for example using OBS), upload it and paste the link here:

* Demo video:
  `https://youtu.be/your_demo_video_id`

---

## Environment Overview

The environment is defined in `environment.py` as `DoublePendulumEnv`, following the Gymnasium API.

### What’s happening inside?

* A physics simulation is created using `pymunk`
* A cart moves horizontally on a track
* Two poles are attached (making it a double inverted pendulum)
* The goal is to keep both poles balanced upright

### Observations (State)

The agent receives 6 values:

* Cart position
* Cart velocity
* Pole 1 angle
* Pole 1 angular velocity
* Pole 2 angle
* Pole 2 angular velocity

### Actions

* A single value between `-1` and `1`
* This controls the force applied to the cart

### Episode Ends When

* Cart goes off track
* Any pole tilts more than 60°
* Maximum steps are reached

---

## Reward Design

There are **two types of rewards** you can use:

### 1. Baseline Reward

This is the simple version:

[
r = \cos(\theta_1) + \cos(\theta_2)
]

* Maximum reward = 2 (both poles upright)
* Encourages balancing only

---

### 2. Shaped Reward

This adds extra penalties for better learning:

[
r = (\cos(\theta_1) + \cos(\theta_2)) - 0.1|x| - 0.01(|\omega_1| + |\omega_2|) - 0.001a^2
]

It penalizes:

* Cart moving too far from center
* Fast swinging of poles
* Large control actions

👉 This helps the agent learn smoother and more stable behavior.

---

## How to Run

### 1. Build Docker

```bash
docker-compose build
```

---

### 2. Run Without Docker (Optional)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -c "from environment import DoublePendulumEnv; env = DoublePendulumEnv(); print(env.observation_space, env.action_space); env.close()"
```

---

### 3. Train the Model

**Baseline reward:**

```bash
docker-compose run train_baseline
```

**Shaped reward:**

```bash
docker-compose run train_shaped
```

Models will be saved in:

* `models/ppo_baseline.zip`
* `models/ppo_shaped.zip`

Logs will be saved in:

* `logs/baseline/`
* `logs/shaped/`

---

### 4. Evaluate the Agent

```bash
docker-compose run evaluate
```

To record a GIF:

```bash
docker-compose run evaluate \
  python evaluate.py \
  --model_path models/ppo_shaped.zip \
  --record_gif True \
  --gif_path media/agent_final.gif
```

---

### 5. Plot Results

After training both models:

```bash
docker-compose run app python plot_rewards.py
```

This generates:

* `reward_comparison.png`

---

## Project Structure

```
.
├── environment.py
├── train.py
├── evaluate.py
├── plot_rewards.py
├── models/
├── logs/
├── media/
├── README.md
```

---

## Environment Variables

You can configure settings using `.env`:

```
DISPLAY=:0
TRAIN_TIMESTEPS=200000
RECORD_GIF=true
```

---

## Notes

* Uses **Gymnasium API** (new format)
* Fixed timestep (`1/60`) ensures stable physics
* Reward shaping improves learning speed and stability

---

## Summary

This project shows how reinforcement learning can solve a **complex control problem** like balancing a double inverted pendulum.

* Baseline reward → simple but slower learning
* Shaped reward → faster and smoother learning

