#!/usr/bin/env python3
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

# -------------------------------------------------
# LOGGING
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)

logging.getLogger("phonemizer").setLevel(logging.ERROR)

# -------------------------------------------------
# LIGHT NORMALIZER (kept)
# -------------------------------------------------
def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.strip()
    text = " ".join(text.split())

    # ultra-cheap punctuation fixes
    text = text.replace(" .", ".")
    text = text.replace(" !", "!")
    text = text.replace(" ?", "?")

    return text


# -------------------------------------------------
# VOICES (UNCHANGED)
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
}

SUPPORTED_LANGS = ["en-us", "en-gb", "ja", "zh-cn", "fr-fr"]


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

    traits = VOICE_TRAITS.get(v_code)
    name = v_code.split("_")[-1].capitalize()

    pretty = f"{name} ({traits['gender']}, {traits['tone']})" if traits else name

    return lang, pretty


# -------------------------------------------------
# WYOMING HANDLER (OPTIMIZED)
# -------------------------------------------------
class KokoroWyomingHandler(AsyncEventHandler):

    def __init__(self, kokoro, default_voice, speed, loop, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kokoro = kokoro
        self.default_voice = default_voice
        self.speed = speed
        self.loop = loop

        # warmup (CRITICAL for HA latency perception)
        try:
            self.kokoro.create("hello", voice=self.default_voice, speed=self.speed, lang="en-us")
        except Exception:
            pass

    async def handle_event(self, event: Event) -> bool:

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

            await self.write_event(Event(type="info", data={
                "tts": [{
                    "name": "kokoro",
                    "description": "Kokoro TTS",
                    "languages": SUPPORTED_LANGS,
                    "installed": True,
                    "voices": voices,
                }]
            }))
            return True

        if event.type == "synthesize":

            synth = Synthesize.from_event(event)
            voice = synth.voice.name if synth.voice else self.default_voice
            lang, _ = resolve_voice(voice)

            raw_text = synth.text
            if not raw_text:
                return True

            clean_text = normalize_text(raw_text)
            if len(clean_text) < 2:
                return True

            start = time.perf_counter()

            try:
                # -------------------------------------------------
                # OFFLOAD BLOCKING TTS TO THREAD (IMPORTANT FIX)
                # -------------------------------------------------
                samples, sr = await self.loop.run_in_executor(
                    None,
                    self.kokoro.create,
                    clean_text,
                    voice,
                    self.speed,
                    lang,
                )

                audio = (samples * 32767).astype("int16").tobytes()

                await self.write_event(AudioStart(rate=sr, width=2, channels=1).event())
                await self.write_event(AudioChunk(audio=audio, rate=sr, width=2, channels=1).event())
                await self.write_event(AudioStop().event())

                _LOGGER.info(
                    "TTS %.3fs | %d chars",
                    time.perf_counter() - start,
                    len(clean_text)
                )

            except Exception as e:
                _LOGGER.exception("TTS failure")
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

    if not onnx_files or not bin_files:
        raise FileNotFoundError("Missing model or voices files")

    model_path = args.model or str(onnx_files[0])
    voices_path = args.voices or str(bin_files[0])

    provider = "CPUExecutionProvider"
    if not args.cpu and "CUDAExecutionProvider" in ort.get_available_providers():
        provider = "CUDAExecutionProvider"

    _LOGGER.info("Provider: %s", provider)

    kokoro = Kokoro(model_path, voices_path)

    server = AsyncServer.from_uri(args.uri)
    loop = asyncio.get_running_loop()

    _LOGGER.info("Listening on %s", args.uri)

    await server.run(
        lambda r, w: KokoroWyomingHandler(
            kokoro, args.voice, args.speed, loop, r, w
        )
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
