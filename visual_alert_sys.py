import time
import threading
import tkinter as tk
from dataclasses import dataclass


@dataclass
class _FlashConfig:
    duration_on:  float = 0.3
    duration_off: float = 0.3
    alpha:        float = 0.6


class VisualAlertSystem:
    def __init__(self, cfg: _FlashConfig | None = None):
        self.cfg = cfg or _FlashConfig()
        self._root: tk.Tk | None = None
        self._flashing = False


    def start_flash(self):
        if self._flashing:
            return
        self._flashing = True
        self._ensure_window()
        self._flash_on()

    def stop_flash(self):
        self._flashing = False
        if self._root is not None:
            self._root.destroy()
            self._root = None



    def _ensure_window(self):
        if self._root is not None:
            return
        self._root           = tk.Tk()
        #no border
        self._root.overrideredirect(True)
        self._root.attributes("-fullscreen", True)
        self._root.attributes("-topmost", True)
        self._root.attributes("-alpha", self.cfg.alpha)
        self._root.configure(background="")
#Esc to cancel
        self._root.bind("<Escape>", lambda e: self.stop_flash())
        threading.Thread(target=self._root.mainloop, daemon=True).start()

    def _flash_on(self):
        if not self._flashing or self._root is None:
            return
        self._root.configure(background="red")
        self._root.after(int(self.cfg.duration_on * 1000), self._flash_off)

    def _flash_off(self):
        if not self._flashing or self._root is None:
            return
        self._root.configure(background="")
        self._root.after(int(self.cfg.duration_off * 1000), self._flash_on)



if __name__ == "__main__":
    vis = VisualAlertSystem()
    print("Flashing for 5 seconds...")
    vis.start_flash()
    time.sleep(5)
    vis.stop_flash()
    print("Done.")
