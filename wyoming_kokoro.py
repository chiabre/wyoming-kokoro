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

SUPPORTED_LANGS = {
    "en-us", "en-gb", "ja", "zh", "fr-fr", "hi", "it", "pt"
}


def resolve_voice(v_code: str):
    """
    Unified resolver for:
    - language mapping
    - metadata naming
    - synthesis routing safety
    """

    lang_map = {
        "af": "en-us",
        "am": "en-us",
        "ef": "en-us",
        "em": "en-us",

        "bf": "en-gb",
        "bm": "en-gb",

        "jf": "ja",
        "jm": "ja",

        "zf": "zh",
        "zm": "zh",

        "ff": "fr-fr",

        "hf": "hi",
        "hm": "hi",

        "if": "it",
        "im": "it",

        "pf": "pt",
        "pm": "pt",
    }

    prefix = v_code[:2]

    lang = lang_map.get(prefix)
    if lang is None:
        _LOGGER.warning("Unknown voice prefix '%s', defaulting to en-us", prefix)
        lang = "en-us"

    if lang not in SUPPORTED_LANGS:
        _LOGGER.warning("Unsupported resolved language %s, forcing en-us", lang)
        lang = "en-us"

    traits = VOICE_TRAITS.get(v_code)

    if traits:
        pretty = f"{v_code.split('_')[-1].capitalize()} ({traits['gender']}, {traits['tone']})"
    else:
        gender = "Female" if "_f_" in v_code or prefix.endswith("f") else "Male"
        pretty = f"{v_code.split('_')[-1].capitalize()} ({gender})"

    return {
        "language": lang,
        "pretty_name": pretty,
    }


class KokoroWyomingHandler(AsyncEventHandler):
    def __init__(self, kokoro, default_voice, speed, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kokoro = kokoro
        self.default_voice = default_voice
        self.speed = speed

    async def handle_event(self, event: Event) -> bool:

        # -------------------------------------------------
        # DESCRIBE
        # -------------------------------------------------
        if event.type == "describe":
            voice_list = []

            for v in self.kokoro.get_voices():
                info = resolve_voice(v)

                voice_list.append({
                    "name": v,
                    "description": info["pretty_name"],
                    "languages": [info["language"]],
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
                    "installed": True,
                    "voices": voice_list,
                }]
            }))
            return True

        # -------------------------------------------------
        # SYNTHESIZE
        # -------------------------------------------------
        if event.type == "synthesize":
            synth = Synthesize.from_event(event)
            voice = synth.voice.name if synth.voice else self.default_voice

            voice_info = resolve_voice(voice)
            engine_lang = voice_info["language"]

            _LOGGER.debug(
                "Synthesizing text='%s' voice=%s lang=%s",
                synth.text, voice, engine_lang
            )

            try:
                start = time.perf_counter()

                samples, sample_rate = self.kokoro.create(
                    synth.text,
                    voice=voice,
                    speed=self.speed,
                    lang=engine_lang,
                )

                _LOGGER.debug("Inference took %.3fs", time.perf_counter() - start)

                audio = (samples * 32767).astype("int16").tobytes()

                await self.write_event(AudioStart(
                    rate=sample_rate, width=2, channels=1
                ).event())

                await self.write_event(AudioChunk(
                    audio=audio, rate=sample_rate, width=2, channels=1
                ).event())

                await self.write_event(AudioStop().event())

            except Exception as e:
                _LOGGER.exception("Synthesis error")
                await self.write_event(Event(type="error", data={"message": str(e)}))
                return False

            return True

        return True


# =========================================================
# MAIN
# =========================================================
async def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data = os.path.join(script_dir, "data")

    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="tcp://0.0.0.0:10200")
    parser.add_argument("--data-dir", default=default_data)
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

    model_path = args.model or str(sorted(data_path.glob("*.onnx"))[0])
    voices_path = args.voices or str(sorted(data_path.glob("*.bin"))[0])

    available = ort.get_available_providers()

    if args.cpu:
        provider = "CPUExecutionProvider"
    elif "CUDAExecutionProvider" in available:
        provider = "CUDAExecutionProvider"
    else:
        provider = "CPUExecutionProvider"

    _LOGGER.info("Provider: %s", provider)
    _LOGGER.info("Model: %s", model_path)
    _LOGGER.info("Voices: %s", voices_path)

    if not Path(model_path).exists():
        raise FileNotFoundError(model_path)
    if not Path(voices_path).exists():
        raise FileNotFoundError(voices_path)

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