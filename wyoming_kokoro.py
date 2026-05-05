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

# Initial logging setup (will be overridden in main)
logging.basicConfig(level=logging.INFO)

_LOGGER = logging.getLogger(__name__)

VOICE_TRAITS = {
    # --------------------
    # English (US)
    # --------------------
    "af_alloy":   {"gender": "female", "lang": "en-us", "overall_grade": "C"},
    "af_aoede":   {"gender": "female", "lang": "en-us", "overall_grade": "C+"},
    "af_bella":   {"gender": "female", "lang": "en-us", "overall_grade": "A-"},
    "af_heart":   {"gender": "female", "lang": "en-us", "overall_grade": "A"},
    "af_jessica": {"gender": "female", "lang": "en-us", "overall_grade": "D"},
    "af_kore":    {"gender": "female", "lang": "en-us", "overall_grade": "C+"},
    "af_nicole":  {"gender": "female", "lang": "en-us", "overall_grade": "B-"},
    "af_nova":    {"gender": "female", "lang": "en-us", "overall_grade": "C"},
    "af_river":   {"gender": "female", "lang": "en-us", "overall_grade": "D"},
    "af_sarah":   {"gender": "female", "lang": "en-us", "overall_grade": "C+"},
    "af_sky":     {"gender": "female", "lang": "en-us", "overall_grade": "C-"},


    "am_adam":    {"gender": "male", "lang": "en-us", "overall_grade": "F+"},
    "am_echo":    {"gender": "male", "lang": "en-us", "overall_grade": "D"},
    "am_eric":    {"gender": "male", "lang": "en-us", "overall_grade": "D"},
    "am_fenrir":  {"gender": "male", "lang": "en-us", "overall_grade": "C+"},
    "am_liam":    {"gender": "male", "lang": "en-us", "overall_grade": "D"},
    "am_michael": {"gender": "male", "lang": "en-us", "overall_grade": "C+"},
    "am_onyx":    {"gender": "male", "lang": "en-us", "overall_grade": "D"},
    "am_puck":    {"gender": "male", "lang": "en-us", "overall_grade": "C+"},
    "am_santa":   {"gender": "male", "lang": "en-us", "overall_grade": "D-"},


    # --------------------
    # English (UK)
    # --------------------
    "bf_alice":    {"gender": "female", "lang": "en-gb", "overall_grade": "D"},
    "bf_emma":     {"gender": "female", "lang": "en-gb", "overall_grade": "B-"},
    "bf_isabella": {"gender": "female", "lang": "en-gb", "overall_grade": "C"},
    "bf_lily":     {"gender": "female", "lang": "en-gb", "overall_grade": "D"},

    "bm_daniel":   {"gender": "male", "lang": "en-gb", "overall_grade": "D"},
    "bm_fable":    {"gender": "male", "lang": "en-gb", "overall_grade": "C"},
    "bm_george":   {"gender": "male", "lang": "en-gb", "overall_grade": "C"},
    "bm_lewis":    {"gender": "male", "lang": "en-gb", "overall_grade": "D+"},


    # --------------------
    # Japanese
    # --------------------
    "jf_alpha":      {"gender": "female", "lang": "ja", "overall_grade": "C+"},
    "jf_gongitsune": {"gender": "female", "lang": "ja", "overall_grade": "C"},
    "jf_nezumi":     {"gender": "female", "lang": "ja", "overall_grade": "C-"},
    "jf_tebukuro":   {"gender": "female", "lang": "ja", "overall_grade": "C"},
    "jm_kumo":       {"gender": "male", "lang": "ja", "overall_grade": "C-"},


    # --------------------
    # Mandarin
    # --------------------
    "zf_xiaobei":  {"gender": "female", "lang": "zh-cn", "overall_grade": "D"},
    "zf_xiaoni":   {"gender": "female", "lang": "zh-cn", "overall_grade": "D"},
    "zf_xiaoxiao": {"gender": "female", "lang": "zh-cn", "overall_grade": "D"},
    "zf_xiaoyi":   {"gender": "female", "lang": "zh-cn", "overall_grade": "D"},

    "zm_yunjian":  {"gender": "male", "lang": "zh-cn", "overall_grade": "D"},
    "zm_yunxi":    {"gender": "male", "lang": "zh-cn", "overall_grade": "D"},
    "zm_yunxia":   {"gender": "male", "lang": "zh-cn", "overall_grade": "D"},
    "zm_yunyang":  {"gender": "male", "lang": "zh-cn", "overall_grade": "D"},


    # --------------------
    # French
    # --------------------
    "ff_siwis": {"gender": "female", "lang": "fr-fr", "overall_grade": "B-"},


    # --------------------
    # Spanish
    # --------------------
    "ef_dora":  {"gender": "female", "lang": "es-es"},
    "em_alex":  {"gender": "male", "lang": "es-es"},
    "em_santa": {"gender": "male", "lang": "es-es"},


    # --------------------
    # Italian
    # --------------------
    "if_sara":   {"gender": "female", "lang": "it-it"},
    "im_nicola": {"gender": "male", "lang": "it-it"},


    # --------------------
    # Hindi
    # --------------------
    "hf_alpha": {"gender": "female", "lang": "hi-in"},
    "hf_beta":  {"gender": "female", "lang": "hi-in"},
    "hm_omega": {"gender": "male", "lang": "hi-in"},
    "hm_psi":   {"gender": "male", "lang": "hi-in"},


    # --------------------
    # Portuguese (BR)
    # --------------------
    "pf_dora":   {"gender": "female", "lang": "pt-br"},
    "pm_alex":   {"gender": "male", "lang": "pt-br"},
    "pm_santa":  {"gender": "male", "lang": "pt-br"},
}

SUPPORTED_LANGS = sorted({
    "en-us",
    "en-gb",
    "ja",
    "zh-cn",
    "fr-fr",
    "es-es",
    "it-it",
    "hi-in",
    "pt-br",
})

def get_voice_metadata(v_code: str):
    traits = VOICE_TRAITS.get(v_code)
    name = v_code.split("_")[-1].capitalize()

    prefix = v_code[:2]

    fallback_map = {
        "af": "en-us",
        "am": "en-us",

        "bf": "en-gb",
        "bm": "en-gb",

        "jf": "ja",
        "jm": "ja",

        "zf": "zh-cn",
        "zm": "zh-cn",

        "ff": "fr-fr",

        "ef": "es-es",
        "em": "es-es",

        "if": "it-it",
        "im": "it-it",

        "hf": "hi-in",
        "hm": "hi-in",

        "pf": "pt-br",
        "pm": "pt-br",
    }

    lang_code = fallback_map.get(prefix, "en-us")

    if lang_code not in SUPPORTED_LANGS:
        lang_code = "en-us"

    if traits:
        grade = traits.get("overall_grade", "N/A")
        pretty_name = f"{name} ({traits['gender']}, {grade})"
    else:
        pretty_name = f"{name}"

    return pretty_name, lang_code

class KokoroWyomingHandler(AsyncEventHandler):
    def __init__(self, kokoro, default_voice, speed, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kokoro = kokoro
        self.default_voice = default_voice
        self.speed = speed

    async def handle_event(self, event: Event) -> bool:
        _LOGGER.debug("Received event: %s", event.type)
        
        if event.type == "describe":
            requested_lang = None
            if hasattr(event, "data") and event.data:
                requested_lang = event.data.get("language")

            voice_list = []

            for v in self.kokoro.get_voices():
                pretty_name, lang_code = get_voice_metadata(v)

                if lang_code is None:
                    continue

                if requested_lang and lang_code != requested_lang:
                    continue

                voice_list.append({
                    "name": v, 
                    "description": pretty_name,
                    "languages": [lang_code], 
                    "installed": True,
                    "attribution": {"name": "hexgrad", "url": "https://github.com/hexgrad/Kokoro-82M"}
                })
            
            await self.write_event(Event(type="info", data={"tts": [{
                "name": "kokoro", 
                "description": "Kokoro TTS",
                "attribution": {"name": "hexgrad", "url": "https://github.com/hexgrad/Kokoro-82M"},
                "installed": True, 
                "voices": voice_list
            }]}))
            return True

        if event.type == "synthesize":
            synth = Synthesize.from_event(event)
            voice = synth.voice.name if synth.voice else self.default_voice

            _, lang_code = get_voice_metadata(voice)
            
            _LOGGER.debug("Synthesizing: '%s' using voice %s (%s)", synth.text, voice, lang_code)
            
            try:
                start_time = time.perf_counter()

                samples, sample_rate = self.kokoro.create(
                    synth.text, 
                    voice=voice, 
                    speed=self.speed, 
                    lang=lang_code
                )

       
                _LOGGER.debug("Inference took %.2f seconds", time.perf_counter() - start_time)

                audio_data = (samples * 32767).astype("int16").tobytes()
                
                await self.write_event(AudioStart(rate=sample_rate, width=2, channels=1).event())
                await self.write_event(AudioChunk(audio=audio_data, rate=sample_rate, width=2, channels=1).event())
                await self.write_event(AudioStop().event())

                _LOGGER.debug("Audio events sent successfully")

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
    parser.add_argument("--model")
    parser.add_argument("--voices")
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    # Apply debug logging level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        _LOGGER.debug("Debug logging enabled")

    # --- Dynamic Discovery Logic ---
    data_path = Path(args.data_dir)

    # 1. Model resolution
    if args.model:
        model_path = args.model
    else:
        onnx_files = sorted(list(data_path.glob("*.onnx")))
        model_path = str(onnx_files[0]) if onnx_files else os.path.join(args.data_dir, "kokoro-v1.0.onnx")

    # 2. Voice resolution
    if args.voices:
        voices_path = args.voices
    else:
        bin_files = sorted(list(data_path.glob("*.bin")))
        voices_path = str(bin_files[0]) if bin_files else os.path.join(args.data_dir, "voices-v1.0.bin")

    # Provider Handling
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
    
    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info(f"Ready. Listening on {args.uri}")
    
    await server.run(lambda r, w: KokoroWyomingHandler(kokoro, args.voice, args.speed, r, w))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
