import time
import threading
import platform
import numpy as np
import pygame


class SoundAlertSystem:

#adjustable parameters
    VOLUME          = 0.7           # 0.0-1.0 % based
    BASE_FREQ       = 440           #unit is Hz
    DURATION        = 1.0          
    MIN_GAP         = 3.0           #chime time gap




    def __init__(self):
        self._last_time = 0.0
        self._playing   = False

        # init pygame mixer in stereo to avoid silent mono issues
        pygame.mixer.init(frequency=44100, size=-16, channels=2)
        pygame.sndarray.use_arraytype("numpy")
        self.pygame = pygame

        print("SoundAlertSystem ready")



    #request a chime
    def trigger_alert(self, severity: float = 1.0) -> bool:



        now = time.time()
        if self._playing or (now - self._last_time) < self.MIN_GAP:
            return False

        #set severity
        severity = max(0.0, min(severity, 1.0))

        #stop calls from blocking
        threading.Thread(
            target=self._play,
            args=(severity,),
            daemon=True,
        ).start()
        return True

    #make and play tone
    def _play(self, severity: float):
        self._playing = True
        try:
            sr  = 44100
            dur = self.DURATION
            #pitch increase as severity of slouch increases
            freq = self.BASE_FREQ * (1 + (severity - 1) * 0.5)

            t    = np.linspace(0, dur, int(sr * dur), endpoint=False)
            wave = np.sin(2 * np.pi * freq * t) * self.VOLUME

            #50ms delay fade in and out
            fade_len  = int(0.05 * sr)
            envelope  = np.linspace(0, 1, fade_len)
            wave[:fade_len]  *= envelope
            wave[-fade_len:] *= envelope[::-1]

            #int16 stereo buffer, had to do this to avoid sound only going to one speaker and had to make it stereo because mono was sometimes silent for some reason
            pcm  = (wave * 32767).astype(np.int16)
            stereo = np.column_stack((pcm, pcm))

            self.pygame.sndarray.make_sound(stereo).play()
            self.pygame.time.delay(int(dur * 1000))

        except Exception as exc:
            print(f"[SoundAlertSystem] tone failed → {exc}")

        finally:
            self._playing   = False
            self._last_time = time.time()



#test
if __name__ == "__main__":
    sa = SoundAlertSystem()
    print("test chime inc")
    time.sleep(1)
    sa.trigger_alert(1.0)
    #keep script alive long enough to hear it
    time.sleep(2) 




