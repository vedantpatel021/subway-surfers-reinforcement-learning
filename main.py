# main.py

import os
import time
import torch
from environment import SubwaySurfersEnv
from agent       import DQNAgent
from frame_stack import FrameStack
from config      import (
    ACTIONS,
    FRAME_STACK_SIZE_DEFAULT,
    MAX_STEPS_PER_EPISODE,
    SAVE_CHECKPOINT_EVERY,
    CHECKPOINT_DIR
)

def main(resume: bool = False, ckpt_name: str = "latest.pt"):
    # Create environment, agent, and frame stack
    env   = SubwaySurfersEnv()
    agent = DQNAgent()
    stack = FrameStack(FRAME_STACK_SIZE_DEFAULT)

    # Resume from a saved checkpoint
    ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
    if resume and os.path.isfile(ckpt_path):
        agent.load(ckpt_name)
        episode = agent.episodes_done   # continue episode count
        total_steps = agent.steps_done  # continue step count
        print(f"[INFO] Resumed from checkpoint '{ckpt_name}' at total_steps = {total_steps}")
    else:
        episode     = 0
        total_steps = 0

    losses  = []        # will store per-step training losses
    rewards = []        # will store per-episode total rewards
    avg100_history = [] # keeps every 100-episode moving average


    try:
        # Training loop (ctrl c to stop)
        while True:
            # Episode setup
            raw = env.reset()           # grab first grayscale frame (H×W)
            stack.reset(raw)            # fill stack with 5 copies of raw
            state = stack.push(raw)     # now state.shape = (5, H, W)
            episode_loss_start = len(losses)

            ep_reward  = 0
            done        = False
            step_in_ep  = 0

            # Step loop
            while not done and step_in_ep < MAX_STEPS_PER_EPISODE:
                action_idx = agent.select_action(state)

                next_raw, reward, done = env.step(ACTIONS[action_idx])
                ep_reward += reward

                next_state = stack.push(next_raw)
                agent.replay_buffer.push(state, action_idx, reward, next_state, done)

                loss = agent.optimize()
                if loss is not None:
                    losses.append(loss)

                # Advance pointers
                state       = next_state
                total_steps += 1
                step_in_ep  += 1

                # Periodic checkpoint
                if total_steps % SAVE_CHECKPOINT_EVERY == 0:
                    agent.save("latest.pt", episodes_done=episode)
                    print(f"[CKPT] Step {total_steps}: checkpoint saved")

            # Episode end
            episode += 1
            rewards.append(ep_reward)

            # Average loss for this episode
            ep_losses = losses[episode_loss_start:]
            if ep_losses:
                avg_loss = sum(ep_losses) / len(ep_losses)
                loss_str = f"{avg_loss:.4f}"
            else:
                loss_str = "N/A"

            print(f"[Episode {episode}] Steps={step_in_ep:<4} "
                f"Reward={ep_reward:<6.1f} ε={agent.epsilon:.3f} "
                f"AvgLoss={loss_str}")

            # Moving 100-episode average reward
            if episode % 100 == 0:
                avg100 = sum(rewards[-100:]) / 100
                avg100_history.append(avg100)
                print(f"           100-ep moving avg reward: {avg100:.1f}")

    except KeyboardInterrupt:
        # Save on ctrl + c
        agent.save("latest_on_interrupt.pt", episodes_done=episode)
        print("\n[CKPT] Interrupted — saved checkpoint → latest_on_interrupt.pt")

        # Print the full 100-episode moving-average history
        if avg100_history:
            print("\n===== 100-Episode Moving-Average Reward History =====")
            for idx, val in enumerate(avg100_history, start=1):
                print(f"{idx:>3}: {val:.1f}")
            print("======================================================")
        else:
            print("\n(No 100-episode averages accumulated yet.)")

        print("[main] Exiting.")

    finally:
        # any cleanup if desired
        pass


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    # start fresh or resume
    main(resume=True, ckpt_name="latest_on_interrupt.pt")
    # main(resume=False)
