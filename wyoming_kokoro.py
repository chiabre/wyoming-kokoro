import argparse
import asyncio
import logging
import os
import time  # Added for timing debug info
from pathlib import Path
import numpy as np
import onnxruntime as ort
from kokoro_onnx import Kokoro
from wyoming.server import AsyncServer, AsyncEventHandler
from wyoming.event import Event
from wyoming.tts import Synthesize
from wyoming.audio import AudioStart, AudioChunk, AudioStop

# 🔥 NEW
import threading

# Initial logging setup (will be overridden in main)
logging.basicConfig(level=logging.INFO)

_LOGGER = logging.getLogger(__name__)

# ... (ALL YOUR EXISTING CONSTANTS UNCHANGED)

# ----------------------------
# FAST CACHE LAYERS
# ----------------------------
VOICE_LANG_CACHE = {
    v: m["lang"]
    for v, m in VOICE_TRAITS.items()
    if "lang" in m
}

VOICE_META_CACHE = {
    v: m
    for v, m in VOICE_TRAITS.items()
}

def fast_chunk(text: str):
    return [t.strip() for t in text.replace("\n", ". ").split(".") if t.strip()]

def get_voice_metadata(v_code: str):
    traits = VOICE_META_CACHE.get(v_code)

    name = v_code.split("_")[-1].capitalize()
    prefix = v_code[:2]

    lang_code = FALLBACK_MAP.get(prefix, "en-us")
    if lang_code not in SUPPORTED_LANGS:
        lang_code = "en-us"

    if traits:
        grade = traits.get("overall_grade", "N/A")
        pretty_name = f"{name} ({traits['gender']}, {grade})"
    else:
        pretty_name = name

    return pretty_name, lang_code


class KokoroWyomingHandler(AsyncEventHandler):
    def __init__(self, kokoro, default_voice, speed, voices, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kokoro = kokoro
        self.default_voice = default_voice
        self.speed = speed
        self.voices = voices

    # 🔥 NEW: true streaming wrapper
    async def _stream_audio(self, text, voice, lang):
        loop = asyncio.get_event_loop()
        queue = asyncio.Queue(maxsize=4)

        def producer():
            try:
                samples, sample_rate = self.kokoro.create(
                    text,
                    voice=voice,
                    speed=self.speed,
                    lang=lang
                )

                # 🔑 ~50ms chunks (good latency vs overhead)
                chunk_size = int(sample_rate * 0.05)

                for i in range(0, len(samples), chunk_size):
                    chunk = samples[i:i + chunk_size]
                    loop.call_soon_threadsafe(queue.put_nowait, (chunk, sample_rate))

                loop.call_soon_threadsafe(queue.put_nowait, None)

            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, e)

        threading.Thread(target=producer, daemon=True).start()

        while True:
            item = await queue.get()

            if item is None:
                break

            if isinstance(item, Exception):
                raise item

            yield item

    async def handle_event(self, event: Event) -> bool:
        _LOGGER.debug("Received event: %s", event.type)
        
        if event.type == "describe":
            requested_lang = None
            if hasattr(event, "data") and event.data:
                requested_lang = event.data.get("language")

            voice_list = []

            for v in self.voices:
                pretty_name, lang_code = get_voice_metadata(v)

                if requested_lang and lang_code != requested_lang:
                    continue

                voice_list.append({
                    "name": v,
                    "description": pretty_name,
                    "languages": [lang_code],
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
                    "attribution": {
                        "name": "hexgrad",
                        "url": "https://github.com/hexgrad/Kokoro-82M"
                    },
                    "voices": voice_list
                }]
            }))

            return True

        if event.type == "synthesize":
            synth = Synthesize.from_event(event)

            voice = synth.voice.name if synth.voice else self.default_voice
            _, lang_code = get_voice_metadata(voice)

            _LOGGER.debug(
                "Synthesizing: '%s' using voice %s (%s)",
                synth.text, voice, lang_code
            )

            try:
                start_time = time.perf_counter()
                await self.write_event(AudioStart(rate=24000, width=2, channels=1).event())

                chunks = fast_chunk(synth.text)

                # 🔥 STREAMING LOOP (replaces old blocking loop)
                for chunk in chunks:
                    async for samples, sample_rate in self._stream_audio(
                        chunk,
                        voice,
                        lang_code
                    ):
                        audio_data = (samples * 32767).astype("int16").tobytes()

                        await self.write_event(AudioChunk(
                            audio=audio_data,
                            rate=sample_rate,
                            width=2,
                            channels=1
                        ).event())

                        # 🔑 prevents burst delivery / improves smoothness
                        await asyncio.sleep(0)

                _LOGGER.debug("Inference took %.2f seconds", time.perf_counter() - start_time)

                await self.write_event(AudioStop().event())

            except Exception as e:
                _LOGGER.error("Synthesis error: %s", e)

            return False

        return True

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
        _LOGGER.debug("Debug logging enabled")

    data_path = Path(args.data_dir)

    model_path = args.model or str(
        next(iter(data_path.glob("*.onnx")), Path(args.data_dir) / "kokoro-v1.0.onnx")
    )

    voices_path = args.voices or str(
        next(iter(data_path.glob("*.bin")), Path(args.data_dir) / "voices-v1.0.bin")
    )

    available = ort.get_available_providers()

    if args.cpu:
        provider = "CPUExecutionProvider"
    elif "CUDAExecutionProvider" in available:
        provider = "CUDAExecutionProvider"
    else:
        provider = "CPUExecutionProvider"

    os.environ["ONNX_PROVIDER"] = provider

    _LOGGER.info(f"Hardware: {provider}")
    _LOGGER.info(f"Model: {model_path}")
    _LOGGER.info(f"Voices: {voices_path}")

    if not os.path.exists(model_path):
        _LOGGER.error(f"Model file not found: {model_path}")
        return

    if not os.path.exists(voices_path):
        _LOGGER.error(f"Voices file not found: {voices_path}")
        return


    kokoro = Kokoro(model_path, voices_path)
    voices = list(kokoro.get_voices())

    try:
        _LOGGER.info("Warming up model...")
        kokoro.create(
            "hello",
            voice=args.voice,
            speed=1.0,
            lang="en-us"
        )
    except Exception:
        pass

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready on %s", args.uri)

    await server.run(
        lambda r, w: KokoroWyomingHandler(
            kokoro,
            args.voice,
            args.speed,
            voices,
            r,
            w
        )
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
