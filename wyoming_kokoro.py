import argparse
import asyncio
import logging
import os
import numpy as np
import onnxruntime as ort  # Added for hardware detection
from kokoro_onnx import Kokoro
from wyoming.server import AsyncServer, AsyncEventHandler
from wyoming.event import Event
from wyoming.tts import Synthesize
from wyoming.audio import AudioStart, AudioChunk, AudioStop

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)

# Comprehensive Voice Mapping
VOICE_TRAITS = {
    # American Female (af_)
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
    
    # American Male (am_)
    "am_adam": {"gender": "Male", "tone": "Deep/Resonant"},
    "am_echo": {"gender": "Male", "tone": "Neutral"},
    "am_eric": {"gender": "Male", "tone": "Expressive"},
    "am_fenrir": {"gender": "Male", "tone": "Deep/Narrator"},
    "am_liam": {"gender": "Male", "tone": "Clean"},
    "am_michael": {"gender": "Male", "tone": "Strong"},
    "am_onyx": {"gender": "Male", "tone": "Bold"},
    "am_puck": {"gender": "Male", "tone": "Youthful"},
    "am_santa": {"gender": "Male", "tone": "Jolly"},

    # British (bf_ / bm_)
    "bf_alice": {"gender": "Female", "tone": "British Crisp"},
    "bf_emma": {"gender": "Female", "tone": "British Gentle"},
    "bf_isabella": {"gender": "Female", "tone": "British Clear"},
    "bf_lily": {"gender": "Female", "tone": "British Sweet"},
    "bm_daniel": {"gender": "Male", "tone": "British Assertive"},
    "bm_fable": {"gender": "Male", "tone": "British Storyteller"},
    "bm_george": {"gender": "Male", "tone": "British Warm"},
    "bm_lewis": {"gender": "Male", "tone": "British Formal"},

    # International (Japanese, Chinese, etc.)
    "jf_alpha": {"gender": "Female", "tone": "Japanese Clear"},
    "zf_xiaoxiao": {"gender": "Female", "tone": "Mandarin Sweet"},
    "ff_siwis": {"gender": "Female", "tone": "French Soft"},
}

def get_voice_metadata(v_code):
    """Parses voice code into Pretty Name and HA Filter Language."""
    name_parts = v_code.split("_")
    short_name = name_parts[-1].capitalize()
    
    # Language Code Mapping for Client-Side Filtering
    lang_map = {
        "af": "en-us", "am": "en-us", # American
        "bf": "en-gb", "bm": "en-gb", # British
        "jf": "ja",    "jm": "ja",    # Japanese
        "zf": "zh",    "zm": "zh",    # Chinese
        "ff": "fr",                   # French
        "hf": "hi",    "hm": "hi",    # Hindi
        "if": "it",    "im": "it",    # Italian
        "pf": "pt",    "pm": "pt",    # Portuguese
        "ef": "en",    "em": "en"     # Generic English
    }
    
    prefix = v_code[:2]
    lang_code = lang_map.get(prefix, "en-us")
    
    traits = VOICE_TRAITS.get(v_code)
    if traits:
        pretty_name = f"{short_name} ({traits['gender']}, {traits['tone']})"
    else:
        gender = "Female" if "_f_" in v_code or prefix.endswith("f") else "Male"
        pretty_name = f"{short_name} ({gender})"

    return pretty_name, lang_code

class KokoroWyomingHandler(AsyncEventHandler):
    def __init__(self, kokoro, default_voice, speed, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kokoro = kokoro
        self.default_voice = default_voice
        self.speed = speed
        _LOGGER.info("Client connected.")

    async def handle_event(self, event: Event) -> bool:
        if event.type == "describe":
            _LOGGER.info("Sending enriched voice list to client.")
            voice_list = []
            for v in self.kokoro.get_voices():
                pretty_name, lang_code = get_voice_metadata(v)
                voice_list.append({
                    "name": v,
                    "description": pretty_name,
                    "languages": [lang_code],
                    "installed": True,
                    "attribution": {"name": "hexgrad", "url": ""}
                })

            await self.write_event(Event(
                type="info",
                data={
                    "tts": [{
                        "name": "kokoro",
                        "description": "Kokoro TTS (Wyoming Server)",
                        "installed": True,
                        "attribution": {"name": "hexgrad", "url": ""},
                        "voices": voice_list
                    }]
                }
            ))
            return True

        if event.type == "synthesize":
            synth = Synthesize.from_event(event)
            voice = synth.voice.name if synth.voice else self.default_voice
            
            # Map the prefix to the correct engine language
            engine_lang = "en-us"
            if voice.startswith("jf"): engine_lang = "ja"
            elif voice.startswith("zf"): engine_lang = "zh"
            elif voice.startswith("ff"): engine_lang = "fr"
            
            _LOGGER.info(f"Synthesizing: {synth.text[:40]}... [{voice} / {engine_lang}]")
            
            try:
                samples, sample_rate = self.kokoro.create(
                    synth.text, voice=voice, speed=self.speed, lang=engine_lang
                )
                audio_data = (samples * 32767).astype("int16").tobytes()
                await self.write_event(AudioStart(rate=sample_rate, width=2, channels=1).event())
                await self.write_event(AudioChunk(audio=audio_data, rate=sample_rate, width=2, channels=1).event())
                await self.write_event(AudioStop().event())
            except Exception as e:
                _LOGGER.error(f"Error during synthesis: {e}")
            
            return False
        return True

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="tcp://0.0.0.0:10200")
    parser.add_argument("--model", default="kokoro-v1.0.onnx")
    parser.add_argument("--voices", default="voices-v1.0.bin")
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    args = parser.parse_args()

    # Hardware Detection Logic
    available_providers = ort.get_available_providers()
    if args.cpu:
        provider = "CPUExecutionProvider"
        _LOGGER.info("Hardware Mode: Forced CPU")
    elif "CUDAExecutionProvider" in available_providers:
        provider = "CUDAExecutionProvider"
        _LOGGER.info("Hardware Mode: NVIDIA GPU (CUDA)")
    else:
        provider = "CPUExecutionProvider"
        _LOGGER.info("Hardware Mode: CPU (Auto-fallback)")

    # Initialize Kokoro with selected provider
    kokoro = Kokoro(args.model, args.voices, provider=provider)
    
    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info(f"Server listening on {args.uri}")
    await server.run(lambda reader, writer: KokoroWyomingHandler(kokoro, args.voice, args.speed, reader, writer))

if __name__ == "__main__":
    asyncio.run(main())
