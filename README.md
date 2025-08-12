# Deep Q-Learning Agent for Subway Surfers (Vision + Keyboard)

I built a Deep Reinforcement Learning (DRL) agent that learns to play Subway Surfers in the browser — specifically on Poki:  
**https://poki.com/en/g/subway-surfers**

Unlike classic RL benchmarks, I do not have access to a game API. The agent learns purely from screen captures (pixels) and controls the game with simulated keyboard input (arrow keys). Training uses a DQN (Deep Q-Network) with a CNN front-end and a replay buffer, plus checkpointing so I can pause/resume anytime.

---

## Table of Contents
- [Key Ideas](#key-ideas)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Configuration](#configuration)
- [Architecture](#architecture)
  - [Environment (Vision + Keyboard)](#environment-vision--keyboard)
  - [Model (CNN → Q-values)](#model-cnn--q-values)
  - [Agent (DQN, epsilon-greedy, checkpoints)](#agent-dqn-epsilon-greedy-checkpoints)
  - [Replay Buffer](#replay-buffer)
- [Training Loop](#training-loop)
- [Game State Detection](#game-state-detection)
- [Running & Resuming](#running--resuming)
- [Tips & Known Quirks](#tips--known-quirks)
- [Troubleshooting](#troubleshooting)
- [Roadmap / Optional Upgrades](#roadmap--optional-upgrades)
- [Final Note](#final-note)

---

## Key Ideas
- **No internal game state.** I learn from pixels only. I grab a rectangular region of the browser, downscale it, convert to grayscale, and stack the last **5** frames so the network sees short-term motion.
- **Control via keyboard.** I send arrow-key presses (left, right, up, down) via `pyautogui`. There's also a `none` no-op action.
- **Reward shaping.** Each live frame yields a small positive reward; dying yields a large negative reward. This is enough to teach "stay alive longer." There is definitely room for improvement when it comes to rewarding and punishing certain behavior, but due to the limited access to game states, I went with a basic step reward and death penalty.
- **Reset.** When a run ends, I click specific screen coordinates to dismiss the "revive" popup and press PLAY. I then wait until the screen looks like "in-game" again before returning the initial state to avoid 1‑step false death/terminal state detections.
- **Checkpointing.** You can stop anytime and resume later with the same network weights, optimizer state, epsilon, and episode counter.

---

## Repository Structure
```
subway_rl_agent/
├── agent.py            # DQNAgent: selects actions, optimizes, saves/loads
├── config.py           # All hyperparameters and screen coords live here
├── environment.py      # Screen capture, key presses, reset(), rewards
├── frame_stack.py      # Fixed-length stack of the last N grayscale frames
├── model.py            # CNN backbone + MLP head that outputs Q-values
├── replay_buffer.py    # Experience replay (uniform sampling)
├── main.py             # Training loop with checkpointing, logging
├── checkpoints/        # Saved .pt files (weights + optimizer + counters)
└── all other files     # Helper files and debugging files
```

---

## Setup
My setup was on Windows (RTX 2060, Python 3.8, Conda). Install dependencies:

```bash
conda create -n subway_rl python=3.8
conda activate subway_rl
pip install -r requirements.txt
```

Make sure Chrome is installed (I only tested on Chrome, but other browsers probably also work). Open the game at the link above, put the game tab in the **top-left**, and adjust zoom (Ctrl + `-`) so the capture region in `config.py` fully covers the play area. If your GPU is powerful enough, don't bother reframing the tab and only worry about adjusting the capture regions in the config file. Test out different setups to see if your GPU seems to be able to handle a normal sized browser tabs.

> ⚠️ On Windows, some tools (e.g., `pyautogui`) may require the console to be run **as Administrator** for global key events; I only simulate keypresses to the focused window (Chrome).

---

## Configuration
All parameters live in `config.py`. The important ones:

### Capture Region (absolute desktop coords)
```python
CAPTURE_REGION = {
    "top": 147, "left": 60, "width": 318, "height": 159
}
```
This is the rectangle of the screen to grab (the game viewport). I keep Chrome in the same place/zoom so these coordinates stay valid.

### Preprocessing Dimensions
```python
TARGET_H = 84
TARGET_W = 168
```
I preserve the game's 16:9 aspect ratio by using **84×168** (instead of squashing to a square). Grayscale + 5‑frame stack → final state shape is **5×84×168**.

### Actions and Timing
```python
ACTIONS = ["left", "right", "up", "down", "none"]
ACTION_DELAY = 0.1  # small sleep after each keypress
```

### Game State Detection
```python
GAME_OVER_PIXEL = (x_rel, y_rel)     # relative to CAPTURE_REGION
GAME_OVER_RGB   = (255, 235, 0)      # expected "live" yellow color
GAME_OVER_TOLERANCE = 7              # small cushion for RGB drift
```
**Note:** `(x_rel, y_rel)` are relative to the capture region, not the full screen:  
`x_rel = abs_x - CAPTURE_REGION["left"]`, `y_rel = abs_y - CAPTURE_REGION["top"]`.

### Rewards & DQN Hyperparameters
```python
# Reward Structure
STEP_REWARD = 0.1      # Small positive reward per frame alive
DYING_REWARD = -200    # Large negative penalty for game over

# DQN Parameters
LEARNING_RATE = 1e-4
GAMMA         = 0.99
EPS_START     = 1.0
EPS_END       = 0.08
EPS_DECAY     = 2.5e-5
BATCH_SIZE    = 32
REPLAY_CAPACITY = 100_000
TARGET_SYNC_EVERY = 5000
```

---

## Architecture

### Environment (Vision + Keyboard)
**Screen capture.** I grab the `CAPTURE_REGION` at ~10–20 Hz using `mss` (fast) and convert to grayscale. I downscale to 84×168 with `cv2.INTER_AREA` for anti-aliased shrinking.

**Game over detection.** With no API access, I monitor hand-picked pixels that are yellow during gameplay and different colors on overlays or death screens. A tolerance value absorbs small color fluctuations.

**Reset logic.** When an episode ends:
1. (Optional small sleep) Let the revive overlay fully appear
2. Double-click the same screen coordinates:
   - 1st click dismisses the "revive?" prompt
   - 2nd click hits **PLAY** on the home screen
3. Wait until the screen shows active gameplay again
4. Return the initial preprocessed frame

This eliminates "every-other episode dies in 1 step" caused by capturing frames before overlays clear.

### Model (CNN → Q-values)
Classic DQN architecture:

```
Input: 5×84×168 (stacked grayscale frames)
↓
Conv2D (8×8, stride=4) + ReLU
↓
Conv2D (4×4, stride=2) + ReLU  
↓
Conv2D (3×3, stride=1) + ReLU
↓
Flatten → FC(512) → FC(|Actions|)
```

I compute the flatten size programmatically by sending a dummy tensor through the conv layers to avoid dimension mismatches when changing input resolution.

### Agent (DQN, epsilon-greedy, checkpoints)
**Action selection.** ε‑greedy with linear decay:
- With probability ε → pick random action (explore)
- Otherwise → `argmax_a Q(s,a)` from policy network (exploit)

ε decays from `EPS_START` to `EPS_END` by `EPS_DECAY` per step, gradually shifting from exploration to exploitation.

**Training.** Compute DQN target `r + γ * max_a' Q_target(s', a')` and minimize MSE loss against `Q_policy(s,a)` using Adam optimizer. Periodically sync policy weights to target network for stability.

**Checkpointing.** Save/restore complete training state:
- Policy & target network weights
- Optimizer state  
- Step counter, epsilon value, and episode counter

Enables seamless pause/resume of training runs.

### Replay Buffer
Uniform sampling from 100k capacity buffer. Stores `(state, action, reward, next_state, done)` transitions with frame stacking applied dynamically during sampling. Returns PyTorch tensors on the configured device (CUDA/CPU).

---

## Training Loop
**High-level flow:**
1. `env.reset()` → capture initial frame, fill frame stack with 5 copies
2. **Episode loop:**
   - Select action using ε‑greedy policy
   - Execute action, capture next frame, compute reward and terminal state
   - Store transition in replay buffer
   - Optimize network on mini-batch if sufficient samples available
3. **Episode end:** Log performance metrics (steps, reward, ε, training loss)
4. **Periodic saving:** Checkpoint every N steps and on Ctrl+C interruption

Track 100‑episode moving average reward to monitor training progress.

---

## Game State Detection
Without API access, I use pixel-based heuristics:
- Identify stable pixel locations that are bright yellow during active gameplay
- Monitor pixels that change color on menus, overlays, or death screens  
- Convert absolute screen coordinates to capture-region-relative coordinates
- Compare captured RGB values against expected colors with small tolerance
- Declare episode termination when pixel checks fail

This approach is sensitive to window position and game theme changes, requiring consistent browser placement and zoom level.

---

## Running & Resuming

### Start Fresh Training
```bash
python main.py  # Uses default config.py parameters
```

### Resume from Checkpoint
Modify `main.py` to:
```python
main(resume=True, ckpt_name="latest_on_interrupt.pt")
```
The agent restores complete training state including weights, optimizer state, epsilon, and episode counters for seamless continuation.

---

## Tips & Known Quirks
- **Chrome window position matters.** `CAPTURE_REGION` uses absolute desktop pixels—keep Chrome fixed at top‑left with consistent zoom level
- **In-game popups.** Double‑clicks or errant mouse input can trigger game menus. Use small but non‑zero action delays
- **Frame rate stability.** Brief sleep after keypresses prevents event flooding and maintains game responsiveness  
- **VRAM usage.** Grayscale + downscaling + batch‑32 is lightweight on RTX 2060. Increase batch size or conv layer width if GPU utilization is low

---

## Troubleshooting

### Episodes ending after 1 step (alternating pattern)
**Cause:** Reset returning frames before overlay clears  
**Fix:** Add sleep before double-click sequence, or poll pixel state until confirmed in-game

### `pyautogui` exceptions or screenshot failures  
**Cause:** Missing dependencies or multi-monitor issues  
**Fix:** Ensure `pyscreeze` and `Pillow` installed; verify capture region on primary display

### CUDA not detected
**Cause:** PyTorch/CUDA version mismatch  
**Fix:** Check PyTorch CUDA compatibility or set `DEVICE = "cpu"` for debugging

---

## Roadmap / Optional Upgrades
- **Double DQN** targets to reduce Q-value over-estimation
- **Dueling DQN** architecture for better value/advantage decomposition  
- **Prioritized Experience Replay** for sampling informative transitions more frequently
- **Polyak averaging** for target network soft updates
- **Evaluation script** for deterministic testing (`ε = 0`) with performance metrics

---

## Final Note
If you clone this repository, maintain consistent Chrome window positioning and zoom level matching your `CAPTURE_REGION` configuration. You should be able to reproduce training results, utilize pause/resume functionality, and observe steady improvement in the 100‑episode moving‑average reward.

**Have fun playing around :)**
