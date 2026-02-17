# 🎙️ Wyoming Kokoro TTS (GPU/CPU)

[Wyoming Protocol](https://github.com/rhasspy/wyoming) server for high-speed local Text-to-Speech using [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) engine optimized for Home Assistant voice pipelines.

## ⚡ Hardware Selection

This repository is hardware-intelligent and distro-aware. During setup, it detects your OS (Debian/Ubuntu) and GPU status to automate the CUDA 12.6 & cuDNN 9.x installation.

| Feature | 🟢 GPU Mode (Recommended) | 🔵 CPU Mode |
| :--- | :--- | :--- |
| **Performance** | Near-instant (sub-300ms latency) | Balanced (~1-3s latency) |
| **Requirements** | NVIDIA GPU + Driver 550+ | Any modern x86_64 / ARM64 CPU |

---

## ⚙️ Installation

### 1. Setup the Environment
The setup script handles the heavy lifting: installing system dependencies, configuring NVIDIA repositories for your specific Linux distro, and creating a virtual environment.

```bash
git clone https://github.com/chiabre/kokoro-wyoming-gpu.git
cd kokoro-wyoming-gpu
chmod +x script/setup
./script/setup
```

> [!IMPORTANT]
> **GPU Users:** After the setup script finishes, you **must** run the command below (or restart your terminal) to activate the new CUDA paths in your current session:
> ```bash
> source ~/.bashrc
> ```

### 2. Download Model Files
Download the essential model and voice database into the project root:

#### 1. Always download the voice metadata first

```bash
wget -O data/voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

#### 2. Download the ONNX model (Choose ONE based on your hardware)

##### OPTION A: For NVIDIA GPU (Recommended) 
P16 is 2x smaller and faster on GPUs with virtually no quality loss.

```bash
wget -O data/kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.fp16.onnx
```

##### OPTION B: For Standard CPU (Balanced)
Standard FP32 model. Best compatibility across all systems.

```bash
wget -O data/kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
```

##### OPTION C: For Low-Power CPU (Raspberry Pi / Thin Clients)
INT8 is the fastest for CPUs but may have slight "robotic" artifacts.

```bash
# wget -O data/kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx
```

## 🚀 Running the Service
### Manual Execution
To start the server manually:

```bash
# Auto-detect hardware, use ./data
./script/run --uri tcp://0.0.0.0:10200

# CPU mode
./script/run --uri tcp://0.0.0.0:10200 --cpu

#Use a custom folder for all model files
./script/run --data-dir /path/to/models
```

### Systemd Deployment
To run this as a persistent background service:

```bash
chmod +x script/install-service
./script/install-service
```

### Manage the service:
- Logs: journalctl -u wyoming-kokoro-asr -f
- Restart: sudo systemctl restart wyoming-kokoro
