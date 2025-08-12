import torch, numpy as np
from agent import DQNAgent

print(torch.__version__)          # 2.2.1+cu118
print(torch.cuda.is_available())  # True (for me)
print(torch.cuda.get_device_name(0))  # NVIDIA GeForce RTX 2060 (for me)

agent = DQNAgent()
dummy = np.zeros((5, 140, 255), dtype=np.uint8)
action = agent.select_action(dummy)
print("Chosen action index:", action)
