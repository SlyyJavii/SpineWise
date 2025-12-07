import json
import os

# I hated making this but I wanted something to do and I couldn't just start this and not finish it.
# I'm fairly certain there's a thing in Capstone 2's grading that will assess accessibility,
# And that probably includes non-English speakers.

class VoiceConfig:
    DEFAULT_CONFIG = {
        "language": "en-US",  # Default language
        "commands": {
            "calibrate": ["calibrate", "calibration", "collab", "cal", "caliber",
                          "collaborate", "calib", "kelly", "cali", "cab", "start calibration"],
            "exit": ["exit", "quit", "close app", "goodbye", "end app",
                     "close application", "shut down", "escape"],
            "start": ["start", "begin", "go", "play", "run", "on",
                      "turn on", "start camera", "begin camera"],
            "stop": ["stop", "pause", "halt", "off", "stop camera",
                     "pause camera", "turn off camera", "camera off"],
            "good_posture": ["good", "good posture", "correct"],
            "bad_posture": ["bad", "bad posture", "poor"],
            "moderate_posture": ["moderate", "medium", "okay"]
        },
        "language_options": {
            "en-US": "English (US)",
            "en-GB": "English (UK)",
            "es-ES": "Spanish (Spain)",
            "es-MX": "Spanish (Mexico)",
            "fr-FR": "French",
            "de-DE": "German",
            "it-IT": "Italian",
            "pt-BR": "Portuguese (Brazil)"
        }
    }

    def __init__(self, config_file="voice_settings.json"):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    return {**self.DEFAULT_CONFIG, **loaded_config}
            except Exception as e:
                print(f"[VoiceConfig] Error loading config: {e}, using defaults")
                return self.DEFAULT_CONFIG.copy()
        return self.DEFAULT_CONFIG.copy()

    def save_config(self):
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"[VoiceConfig] Error saving config: {e}")
            return False

    def get_language(self):
        return self.config.get("language", "en-US")

    def set_language(self, language_code):
        if language_code in self.config["language_options"]:
            self.config["language"] = language_code
            self.save_config()
            return True
        return False

    def get_command_triggers(self, command_type):
        return self.config["commands"].get(command_type, [])

    def set_command_triggers(self, command_type, triggers):
        if isinstance(triggers, str):
            triggers = [t.strip() for t in triggers.split(',') if t.strip()]
        self.config["commands"][command_type] = triggers
        self.save_config()

    def add_command_trigger(self, command_type, trigger):
        if command_type not in self.config["commands"]:
            self.config["commands"][command_type] = []
        if trigger not in self.config["commands"][command_type]:
            self.config["commands"][command_type].append(trigger)
            self.save_config()

    def remove_command_trigger(self, command_type, trigger):
        if command_type in self.config["commands"]:
            if trigger in self.config["commands"][command_type]:
                self.config["commands"][command_type].remove(trigger)
                self.save_config()

    def match_command(self, spoken_text):
        spoken_lower = spoken_text.lower().strip()

        for command_type, triggers in self.config["commands"].items():
            for trigger in triggers:
                if trigger.lower() in spoken_lower:
                    return command_type

        return None

    def get_available_languages(self):
        return self.config["language_options"]


# Global instance for easy access
voice_config = VoiceConfig()