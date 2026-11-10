import os
import re


_DISPLAY_PUNCTUATION = re.compile(r"""[，。！？；：、,.!?;:"'`~@#$%^&*()\-_=+\[\]{}<>|\\/…—–\r\n]+""")
_BLUETOOTH_INPUT = re.compile(r"bluetooth|hands[\s-]?free|headset|airpods|耳机", re.IGNORECASE)
_PREFERRED_INPUT = re.compile(r"microphone|mic|array|麦克风|阵列|usb", re.IGNORECASE)


def sanitize_display_text(value):
    """Use spaces as the only visible pause separator for ASR output."""
    return re.sub(r"\s+", " ", _DISPLAY_PUNCTUATION.sub(" ", str(value or ""))).strip()


def choose_input_device(devices, default_index=None):
    """Avoid Bluetooth hands-free input when a local microphone is available."""
    override = os.environ.get("THINKINGEARS_INPUT_DEVICE", "").strip()
    if override:
        for index, device in enumerate(devices):
            name = str(device.get("name", ""))
            if override == str(index) or override.lower() in name.lower():
                return index, name

    def usable(index, device):
        return int(device.get("max_input_channels", 0) or 0) > 0 and not _BLUETOOTH_INPUT.search(str(device.get("name", "")))

    if isinstance(default_index, int) and 0 <= default_index < len(devices) and usable(default_index, devices[default_index]):
        return default_index, str(devices[default_index].get("name", ""))

    candidates = [(index, device) for index, device in enumerate(devices) if usable(index, device)]
    if candidates:
        index, device = max(
            candidates,
            key=lambda item: (
                bool(_PREFERRED_INPUT.search(str(item[1].get("name", "")))),
                int(item[1].get("max_input_channels", 0) or 0),
                float(item[1].get("default_samplerate", 0) or 0),
            ),
        )
        return index, str(device.get("name", ""))

    if isinstance(default_index, int) and 0 <= default_index < len(devices):
        return default_index, str(devices[default_index].get("name", ""))
    for index, device in enumerate(devices):
        if int(device.get("max_input_channels", 0) or 0) > 0:
            return index, str(device.get("name", ""))
    return None, ""
