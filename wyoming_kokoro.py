import argparse
import asyncio
import logging
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from kokoro_onnx import Kokoro

from wyoming.server import AsyncServer, AsyncEventHandler
from wyoming.event import Event
from wyoming.tts import Synthesize
from wyoming.audio import AudioStart, AudioChunk, AudioStop

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)

# -------------------------------------------------
# VOICE METADATA
# -------------------------------------------------
VOICE_TRAITS = {
    "af_alloy": {"gender": "Female", "tone": "Neutral"},
    "af_aoede": {"gender": "Female", "tone": "Lyric"},
    "af_bella": {"gender": "Female", "tone": "Soft/Warm"},
    "af_heart": {"gender": "Female", "tone": "Balanced"},
    "af_jessica": {"gender": "Female", "tone": "Bright"},
    "af_kore": {"gender": "Female", "tone": "Calm"},
    "af_nicole": {"gender": "Female", "tone": "Professional"},
    "af_nova": {"gender": "Female", "tone": "Energetic"},
    "af_river": {"gender": "Female", "tone": "Smooth"},
    "af_sarah": {"gender": "Female", "tone": "Cheerful"},
    "af_sky": {"gender": "Female", "tone": "Friendly"},

    "am_adam": {"gender": "Male", "tone": "Deep/Resonant"},
    "am_echo": {"gender": "Male", "tone": "Neutral"},
    "am_eric": {"gender": "Male", "tone": "Expressive"},
    "am_fenrir": {"gender": "Male", "tone": "Deep/Narrator"},
    "am_liam": {"gender": "Male", "tone": "Clean"},
    "am_michael": {"gender": "Male", "tone": "Strong"},
    "am_onyx": {"gender": "Male", "tone": "Bold"},
    "am_puck": {"gender": "Male", "tone": "Youthful"},
    "am_santa": {"gender": "Male", "tone": "Jolly"},

    "bf_alice": {"gender": "Female", "tone": "British Crisp"},
    "bf_emma": {"gender": "Female", "tone": "British Gentle"},
    "bf_isabella": {"gender": "Female", "tone": "British Clear"},
    "bf_lily": {"gender": "Female", "tone": "British Sweet"},

    "bm_daniel": {"gender": "Male", "tone": "British Assertive"},
    "bm_fable": {"gender": "Male", "tone": "British Storyteller"},
    "bm_george": {"gender": "Male", "tone": "British Warm"},
    "bm_lewis": {"gender": "Male", "tone": "British Formal"},

    "jf_alpha": {"gender": "Female", "tone": "Japanese Clear"},
    "zf_xiaoxiao": {"gender": "Female", "tone": "Mandarin Sweet"},
    "ff_siwis": {"gender": "Female", "tone": "French Soft"},
}

# -------------------------------------------------
# GLOBAL LANGUAGE SET (IMPORTANT FOR HOME ASSISTANT)
# -------------------------------------------------
SUPPORTED_LANGS = sorted({
    "en-us",
    "en-gb",
    "ja",
    "zh-cn",
    "fr-fr",
})

def resolve_voice(v_code: str):
    lang_map = {
        "af": "en-us",
        "am": "en-us",

        "bf": "en-gb",
        "bm": "en-gb",

        "jf": "ja",
        "jm": "ja",

        "zf": "zh-cn",
        "zm": "zh-cn",

        "ff": "fr-fr",
    }

    prefix = v_code[:2]
    lang = lang_map.get(prefix, "en-us")

    # strict normalization (IMPORTANT for HA)
    if lang not in SUPPORTED_LANGS:
        lang = "en-us"

    traits = VOICE_TRAITS.get(v_code)
    name = v_code.split("_")[-1].capitalize()

    if traits:
        pretty = f"{name} ({traits['gender']}, {traits['tone']})"
    else:
        gender = "Female" if "_f_" in v_code or prefix.endswith("f") else "Male"
        pretty = f"{name} ({gender})"

    return lang, pretty


# -------------------------------------------------
# WYOMING HANDLER
# -------------------------------------------------
class KokoroWyomingHandler(AsyncEventHandler):

    def __init__(self, kokoro, default_voice, speed, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kokoro = kokoro
        self.default_voice = default_voice
        self.speed = speed

    async def handle_event(self, event: Event) -> bool:

        # -------------------------
        # DESCRIBE (FIXED FOR HA)
        # -------------------------
        if event.type == "describe":
            voices = []

            for v in self.kokoro.get_voices():
                lang, pretty = resolve_voice(v)

                voices.append({
                    "name": v,
                    "description": pretty,
                    "languages": [lang],
                    "installed": True,
                    "attribution": {
                        "name": "hexgrad",
                        "url": "https://github.com/hexgrad/Kokoro-82M"
                    }
                })

            # IMPORTANT: include global language list
            await self.write_event(Event(type="info", data={
                "tts": [{
                    "name": "kokoro",
                    "description": "Kokoro TTS",

                    # 🔥 THIS IS WHAT FIXES HA LANGUAGE FILTERING
                    "languages": SUPPORTED_LANGS,

                    "installed": True,
                    "voices": voices,

                    "attribution": {
                        "name": "hexgrad",
                        "url": "https://github.com/hexgrad/Kokoro-82M"
                    }
                }]
            }))
            return True

        # -------------------------
        # SYNTHESIZE
        # -------------------------
        if event.type == "synthesize":
            synth = Synthesize.from_event(event)
            voice = synth.voice.name if synth.voice else self.default_voice

            lang, _ = resolve_voice(voice)

            try:
                start = time.perf_counter()

                samples, sr = self.kokoro.create(
                    synth.text,
                    voice=voice,
                    speed=self.speed,
                    lang=lang,
                )

                _LOGGER.debug("Inference %.3fs", time.perf_counter() - start)

                audio = (samples * 32767).astype("int16").tobytes()

                await self.write_event(AudioStart(rate=sr, width=2, channels=1).event())
                await self.write_event(AudioChunk(audio=audio, rate=sr, width=2, channels=1).event())
                await self.write_event(AudioStop().event())

            except Exception as e:
                _LOGGER.exception("Synthesis error")
                await self.write_event(Event(type="error", data={"message": str(e)}))
                return False

            return True

        return True


# -------------------------------------------------
# MAIN
# -------------------------------------------------
async def main():

    base = Path(__file__).parent
    data_dir = base / "data"

    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="tcp://0.0.0.0:10200")
    parser.add_argument("--data-dir", default=str(data_dir))
    parser.add_argument("--model")
    parser.add_argument("--voices")
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    data_path = Path(args.data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    onnx_files = sorted(data_path.glob("*.onnx"))
    bin_files = sorted(data_path.glob("*.bin"))

    if not onnx_files:
        raise FileNotFoundError("No ONNX model found")

    if not bin_files:
        raise FileNotFoundError("No voices file found")

    model_path = args.model or str(onnx_files[0])
    voices_path = args.voices or str(bin_files[0])

    providers = ort.get_available_providers()

    if args.cpu:
        provider = "CPUExecutionProvider"
    elif "CUDAExecutionProvider" in providers:
        provider = "CUDAExecutionProvider"
    else:
        provider = "CPUExecutionProvider"

    _LOGGER.info("Provider: %s", provider)
    _LOGGER.info("Model: %s", model_path)
    _LOGGER.info("Voices: %s", voices_path)

    kokoro = Kokoro(model_path, voices_path)

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Listening on %s", args.uri)

    await server.run(
        lambda r, w: KokoroWyomingHandler(
            kokoro, args.voice, args.speed, r, w
        )
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass