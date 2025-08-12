# Screen capture settings
CAPTURE_REGION = {
    "top": 147,
    "left": 60,
    "width": 318,
    "height": 159
}

# Down-scaled frame size (must match environment & model)
TARGET_H = 84
TARGET_W = 168

FRAME_DELAY = 0.1

# Actions
ACTIONS = ["left", "right", "up", "down", "none"]
ACTION_KEY_MAPPING = {
    "left": "left",
    "right": "right",
    "up": "up",
    "down": "down",
    "none": None
}

# Time delay between actions (seconds)
ACTION_DELAY = 0.1

# Game over detection (pixel coordinates within screen region)


GAME_OVER_PIXEL = (370 - CAPTURE_REGION["left"], 
                   157 - CAPTURE_REGION["top"])     # Coordinate for checking terminal state
GAME_OVER_RGB = (255, 235, 0)                      # RGB value when game ends
HOVERBOARD_PIXEL = (230 - CAPTURE_REGION["left"], 
                     239 - CAPTURE_REGION["top"])
HOVERBOARD_PIXEL_RGB = (51, 157, 22)
GAME_OVER_PIXEL_3 = (332 - CAPTURE_REGION["left"], 
                     141 - CAPTURE_REGION["top"])
GAME_OVER_RGB_3 = (255, 221, 0)
GAME_OVER_TOLERANCE = 7                            # Tolerance in RGB matching

# Game reset location
PLAY_BUTTON = (245, 295)
CLOSE_HOVERBOARD = (172, 170)

# Resting position
MOUSE_RESTING_POSITION = (220, 185)

# BUFFER
FRAME_STACK_SIZE_DEFAULT = 5

# Test Env
SAVE_IMAGE_CAPTURES = True

# REWARD
STEP_REWARD = 0.1
DYING_REWARD = -200

# DQN hyper-parameters
LEARNING_RATE      = 1e-4
GAMMA              = 0.99
EPS_START          = 1.0
EPS_END            = 0.001
EPS_DECAY          = 1e-5        # epsilon decreases by this per step
TARGET_SYNC_EVERY  = 5000        # steps
BATCH_SIZE         = 32
REPLAY_CAPACITY    = 100_000
DEVICE             = "cuda"      # or "cpu"
CHECKPOINT_DIR     = "checkpoints"

# how many frames max per episode (debug safety)
MAX_STEPS_PER_EPISODE  = 10_000

# save every N global steps
SAVE_CHECKPOINT_EVERY  = 1_000