# 🎙️ Wyoming Kokoro TTS (GPU/CPU)

[Wyoming Protocol](https://github.com/rhasspy/wyoming) server for high-speed local Text-to-Speech using [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) engine optimized for Home Assistant voice pipelines.

## 💡 Why Kokoro:
1.  **Memory Efficiency:** While models like *VibeVoice* or *XTTS-v2* require 2GB to 4GB of VRAM, Kokoro fits comfortably in 350MB, making it possible to run an LLM, an ASR model, and TTS all on a single 8GB consumer card or even a high-end laptop CPU.
2.  **Sample Rate:** It outputs natively at **24kHz**, which provides a "crisper" sound than the 16kHz standard used by many older fast models like Piper.
3.  **Zero-Shot Style:** Although it doesn't do "voice cloning" in the traditional sense, you can pass a **Style Vector** to the ONNX session to slightly tweak the emotion or tone without retraining.

## ⚡ Hardware Selection

This repository is hardware-intelligent and distro-aware. During setup, it detects your OS (Debian/Ubuntu) and GPU status to automate the CUDA 12.6 & cuDNN 9.x installation.

| Feature | 🟢 GPU Mode (Recommended) | 🔵 CPU Mode |
| :--- | :--- | :--- |
| **Performance** | Near-instant (sub-300ms latency) | Balanced (~1-3s latency) |
| **Requirements** | NVIDIA GPU + Driver 550+ | Any modern x86_64 / ARM64 CPU |

## ⚙️ Installation

### 1. Setup the Environment
The setup script handles the heavy lifting: installing system dependencies, configuring NVIDIA repositories for your specific Linux distro, and creating a virtual environment.

```bash
git clone https://github.com/chiabre/wyoming-kokoro.git
cd wyoming-kokoro
chmod +x script/setup
./script/setup
```

> [!IMPORTANT]
> **GPU Users:** After the setup script finishes, you **must** run the command below (or restart your terminal) to activate the new CUDA paths in your current session:
> ```bash
> source ~/.bashrc
> ```

### 2. Models

Kokoro-82M family. For the best balance of speed and quality use **v1.0 FP16** model.

> [!IMPORTANT]
> Always ensure the model version (e.g., v1.0) matches the voices version. Using v1.0 models with v0.19 voices will result in corrupted audio output.

| Model Name | Version | Precision | RAM | Description | Filename |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Kokoro-82M | v1.0 | Int8 | ~350MB | **Fastest.** Best for Raspberry Pi/CPUs. | `kokoro-v1.0.int8.onnx` |
| Kokoro-82M | v1.0 | FP16 | ~500MB | **Recommended.** Best for NVIDIA GPUs. | `kokoro-v1.0.fp16.onnx` |
| Kokoro-82M | v1.0 | FP32 | ~800MB | **Original.** Highest compatibility. | `kokoro-v1.0.onnx` |

#### 1.Download Instructions

```bash
# 1. Always download the matching voices file
wget -P data/ https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

# 2. Choose ONE model based on your hardware

# For Low-Power CPU / Raspberry Pi (Int8)
wget -P data/ https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx

# For NVIDIA GPU (FP16 - Recommended)
wget -P data/ https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.fp16.onnx

# For Standard CPU (FP32)
wget -P data/ https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
```

> [!IMPORTANT]
> To select a specific model when multiple files are present in the data directory, use the `--model` parameter.

## 🚀 Running the Service
### Manual Execution
To start the server manually:

```bash
# Standard auto-detection (uses files found in ./data)
python3 -m wyoming_kokoro

# Explicitly use the high-speed FP16 model
python3 -m wyoming_kokoro --model data/kokoro-v1.0.fp16.onnx

# Force CPU mode for low-power devices
python3 -m wyoming_kokoro --cpu --speed 1.1
```
#### ⚙️ Configuration Options
These parameters match the latest wyoming-kokoro implementation. If no model or voices file is specified, the server will auto-detect the first available files in your --data-dir.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--uri` | Optional | `tcp://0.0.0.0:10200` | The address and port for the Wyoming server. |
| `--data-dir` | Optional | `./data` | Where to look for `.onnx` and `.bin` files. |
| `--model` | Optional | *Auto-detect* | Explicit path to the `.onnx` Kokoro ONNX model.  |
| `--voices` | Optional | *Auto-detect* | Explicit path to the `voices.bin` file.  |
| `--voice` | Optional | `af_heart` | Default voice if none is requested by the client. |
| `--speed` | Optional | `1.0` | Global speed multiplier for speech synthesis. |
| `--cpu` | Flag | `False` | Force CPU inference even if a GPU is detected. |

> [!IMPORTANT]
> The server *Auto-detect* for model and voices selects the first `.onnx` and `.bin` files it finds in the data directory by sorting all matches alphabetically.

### Systemd Deployment
To run this as a persistent background service:

```bash
chmod +x script/install-service
./script/install-service
```

### Manage the service:
- Logs: journalctl -u wyoming-kokoro-asr -f
- Restart: sudo systemctl restart wyoming-kokoro
