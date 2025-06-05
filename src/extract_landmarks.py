"""Example script determining posture and showing notifications."""

import random
import time

from visual_alert_sys import show_notification


def posture_is_bad() -> bool:
    """Placeholder function that randomly determines if posture is bad."""
    return random.random() < 0.3


def main() -> None:
    """Continuously check posture and notify if bad."""
    try:
        while True:
            if posture_is_bad():
                show_notification("Bad Posture Detected", "Please adjust your posture.")
            time.sleep(5)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
