import argparse
import asyncio
import logging
import os
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

def get_voice_metadata(v_code):
    name_parts = v_code.split("_")
    short_name = name_parts[-1].capitalize()
    lang_map = {
        "af": "en-us", "am": "en-us", "bf": "en-gb", "bm": "en-gb",
        "jf": "ja", "jm": "ja", "zf": "zh", "zm": "zh", "ff": "fr",
        "hf": "hi", "hm": "hi", "if": "it", "im": "it", "pf": "pt",
        "pm": "pt", "ef": "en", "em": "en"
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

    async def handle_event(self, event: Event) -> bool:
        if event.type == "describe":
            voice_list = []
            for v in self.kokoro.get_voices():
                pretty_name, lang_code = get_voice_metadata(v)
                voice_list.append({
                    "name": v, "description": pretty_name,
                    "languages": [lang_code], "installed": True,
                    "attribution": {"name": "hexgrad", "url": ""}
                })
            await self.write_event(Event(type="info", data={"tts": [{
                "name": "kokoro", "description": "Kokoro TTS",
                "installed": True, "voices": voice_list
            }]}))
            return True

        if event.type == "synthesize":
            synth = Synthesize.from_event(event)
            voice = synth.voice.name if synth.voice else self.default_voice
            
            # Map language for the engine
            engine_lang = "en-us"
            if voice.startswith("jf"): engine_lang = "ja"
            elif voice.startswith("zf"): engine_lang = "zh"
            elif voice.startswith("ff"): engine_lang = "fr"
            
            try:
                # create() returns (samples, sample_rate)
                samples, sample_rate = self.kokoro.create(
                    synth.text, voice=voice, speed=self.speed, lang=engine_lang
                )
                # Convert float32 samples to int16 PCM
                audio_data = (samples * 32767).astype("int16").tobytes()
                
                await self.write_event(AudioStart(rate=sample_rate, width=2, channels=1).event())
                await self.write_event(AudioChunk(audio=audio_data, rate=sample_rate, width=2, channels=1).event())
                await self.write_event(AudioStop().event())
            except Exception as e:
                _LOGGER.error(f"Synthesis error: {e}")
            return False
        return True

async def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data = os.path.join(script_dir, "data")

    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="tcp://0.0.0.0:10200")
    parser.add_argument("--data-dir", default=default_data)
    parser.add_argument("--model", help="Path to ONNX file")
    parser.add_argument("--voices", help="Path to voices.bin")
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    # Model path logic
    model_path = args.model or os.path.join(args.data_dir, "kokoro-v1.0.onnx")
    
    # Voice path logic: Try common names if not provided
    if args.voices:
        voices_path = args.voices
    else:
        v1_path = os.path.join(args.data_dir, "voices-v1.0.bin")
        v_simple_path = os.path.join(args.data_dir, "voices.bin")
        voices_path = v1_path if os.path.exists(v1_path) else v_simple_path

    # Provider Handling for v0.5.0 via environment variables
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

    # Kokoro __init__ no longer takes 'provider' keyword in v0.5.0
    kokoro = Kokoro(model_path, voices_path)
    
    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info(f"Ready. Listening on {args.uri}")
    
    await server.run(lambda r, w: KokoroWyomingHandler(kokoro, args.voice, args.speed, r, w))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
