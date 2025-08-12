
import os
import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from model import DQN
from replay_buffer import ReplayBuffer
from frame_stack import FrameStack
from config import *


class DQNAgent:

    def __init__(self, state_shape: Tuple[int, int, int] = (FRAME_STACK_SIZE_DEFAULT, 140, 255), num_actions: int = len(ACTIONS)):
        in_channels, H, W = state_shape

        # Policy network (the one we learn)
        self.policy_net = DQN(in_channels, num_actions).to(DEVICE)

        # Target network (slow-moving copy)
        self.target_net = DQN(in_channels, num_actions).to(DEVICE)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() #inference only

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_CAPACITY)

        # Frame stacker (building the 5 frame states)
        self.frame_stack = FrameStack(FRAME_STACK_SIZE_DEFAULT)

        # Epsilon-greedy tracking
        self.steps_done = 0
        self.epsilon = EPS_START

        # Episode tracking
        self.episodes_done = 0

        # For checkpointing
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)


    def select_action(self, state_stack: np.ndarray) -> int:
        # Decay epsilon
        self.epsilon = max(EPS_END, EPS_START - self.steps_done * EPS_DECAY)
        self.steps_done += 1

        if random.random() < self.epsilon:
            # explore
            return random.randrange(len(ACTIONS))
        else:
            # exploit and feed through policy net
            state_t = (torch.from_numpy(state_stack).unsqueeze(0).float() / 255.0)
            state_t = state_t.to(DEVICE)
            with torch.no_grad():
                q_values = self.policy_net(state_t)
            return int(q_values.argmax(dim=1).item())
        
    
    def optimize(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return None  # not enough samples yet—no loss to report
        
        (state_b, action_b, reward_b, next_state_b, done_b) = self.replay_buffer.sample(BATCH_SIZE)

        # to torch
        state_b = torch.tensor(state_b, dtype=torch.float32, device=DEVICE) / 255.0
        action_b = torch.tensor(action_b, dtype=torch.long,   device=DEVICE)
        reward_b = torch.tensor(reward_b, dtype=torch.float32, device=DEVICE)
        next_state_b = torch.tensor(next_state_b, dtype=torch.float32, device=DEVICE) / 255.0
        done_b = torch.tensor(done_b,   dtype=torch.float32, device=DEVICE)

        # Current Q(s,a) -> gather along action dim
        q_pred = self.policy_net(state_b).gather(1, action_b.unsqueeze(1)).squeeze(1)

        # Target Q(s', a') using target network
        with torch.no_grad():
            q_next = self.target_net(next_state_b).max(1)[0]
            q_target = reward_b + (1.0 - done_b) * GAMMA * q_next
        
        # Loss
        loss = nn.functional.mse_loss(q_pred, q_target)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodically sync target network
        if self.steps_done % TARGET_SYNC_EVERY == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()

    def save(self, fname: str, episodes_done: int = 0):
        checkpoint = {
            "policy_state":  self.policy_net.state_dict(),
            "target_state":  self.target_net.state_dict(),
            "optimizer":     self.optimizer.state_dict(),
            "steps_done":    self.steps_done,
            "epsilon":       self.epsilon,
            "episodes_done": episodes_done,
        }
        torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, fname))

    def load(self, fname: str):
        path = os.path.join(CHECKPOINT_DIR, fname)
        ckpt = torch.load(path, map_location=DEVICE)
        self.policy_net.load_state_dict(ckpt["policy_state"])
        self.target_net.load_state_dict(ckpt["target_state"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps_done = ckpt["steps_done"]
        self.epsilon = ckpt["epsilon"]
        self.episodes_done = ckpt.get("episodes_done", 0)