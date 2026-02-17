# 🎙️ Wyoming Kokoro TTS (GPU/CPU)

[Wyoming Protocol](https://github.com/rhasspy/wyoming) server for the [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) TTS engine, featuring automatic NVIDIA GPU acceleration / CPU fallback.

## ⚡ Hardware Selection

This repository is hardware-intelligent. It detects your environment and optimizes the engine accordingly.

| Feature | 🟢 GPU Mode (Recommended) | 🔵 CPU Mode |
| :--- | :--- | :--- |
| **Performance** | Near-instant (sub-300ms latency) | Balanced (~1-3s latency) |
| **Requirements** | NVIDIA GPU + Driver 550+ | Any modern x86_64 / ARM64 CPU |
| **Libraries** | CUDA 12.x & cuDNN 9.x | No extra drivers required |

---

## ⚙️ Installation

### 1. Setup the Environment
Clone the repo and run the setup script. It will detect your hardware and install either `onnxruntime-gpu` or the `standard onnxruntime`.

```bash
git clone [https://github.com/your-username/kokoro-wyoming-gpu.git](https://github.com/your-username/kokoro-wyoming-gpu.git)
cd kokoro-wyoming-gpu
chmod +x script/setup
./script/setup
```

### 2. Download Model Files
Download the essential model and voice database into the project root:

```bash
# Download the Kokoro ONNX model
wget -O data/kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx

# Download the voice metadata
wget -O data/voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
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
