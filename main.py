import pygame
# Imports the Pygame library for sound
import os
print(">>> Script started")
print("Working directory:", os.getcwd())
import time
# Imports the time module required for the sleep method to work
pygame.init()
#Initializes all the pygame modules. We always need to call this when playing with pygame
pygame.mixer.init()
#The mixer  module handles anything related to sound. Without this, Pygame cannot load or play any sound
beep = pygame.mixer.Sound("bad_posture_alert.wav")
#A constructor that returns a Sound object that we can: play, stop, loop, or control the volume for.
#The argument "bad_posture_alert.wav" is the name of the wav file.This MUST be in the folder of the project

#Simulated posture detection variable, only for testing purposes
test_posture_confidence = True


start_time = None #When the user first entered "bad posture."
loop_started = False #This keeps track of whether we've already started the alert beeping loop
last_beep_time = 0 # This tracks the last time a beep was played, so we can create a 2-second gap between each beep. 

while True:
    if test_posture_confidence:
        if start_time is None:
            start_time = time.time()  # Start timing posture
        else:
            elapsed = time.time() - start_time
            if elapsed >= 10 and not loop_started:
                print("Bad posture confirmed for 10 seconds. Starting beeps.")
                loop_started = True  # Activate beeping mode

        # Beep only if alert has started AND 2 seconds passed
        if loop_started and time.time() - last_beep_time >= 2:
            beep.play()
            last_beep_time = time.time()

    else:
        # Reset everything if posture is corrected
        if loop_started:
            beep.stop()
            print("Posture corrected. Stopped alert.")
        start_time = None
        loop_started = False

    time.sleep(0.1)  # Reduce CPU usage


  
