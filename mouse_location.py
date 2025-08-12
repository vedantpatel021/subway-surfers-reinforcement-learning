import pyautogui
import time

print("Move your mouse around. Press Ctrl+C to stop.\n")

try:
    while True:
        x, y = pyautogui.position()
        r, g, b = pyautogui.screenshot().getpixel((x, y))
        print(f"X: {x}, Y: {y} => RGB: ({r}, {g}, {b})", end="\r")
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nStopped.")