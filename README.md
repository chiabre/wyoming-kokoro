# 🎙️ Wyoming Kokoro TTS (GPU/CPU)

[Wyoming Protocol](https://github.com/rhasspy/wyoming) server for the [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) TTS engine, featuring automatic NVIDIA GPU acceleration / CPU fallback.

## ⚡ Hardware Selection

### 🟢 GPU Mode (Recommended)
**Performance:** Near-instant speech generation (sub-300ms latency).  
**Requirements:**
* **NVIDIA Driver:** 550+
* **CUDA:** 12.x
* **cuDNN:** 9.x
* **LXC Users:** Ensure GPU passthrough is configured in your Proxmox lxc.

### 🔵 CPU Mode
**Performance:** Balanced (approx. 1-3s latency).  
**Requirements:**
* **Processor:** Any modern x86_64 or ARM64 CPU.

---

## 🛠 Installation

### 1. Setup the Environment
Clone the repo and run the hardware-aware setup script. It will automatically detect if you have an NVIDIA GPU and install the correct `onnxruntime` package.

```bash
git clone [https://github.com/your-username/kokoro-wyoming-gpu.git](https://github.com/your-username/kokoro-wyoming-gpu.git)
cd kokoro-wyoming-gpu
chmod +x script/setup
./script/setup
```

### 2. Download Model Files
Run these commands from the root of the project folder to download the essential model and voice database:

```bash
# Download the Kokoro ONNX model
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx

# Download the voice metadata
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

## 🚀 Running the Service
### Manual Execution
To start the server manually:

```bash
# GPU / Auto-detect
./script/run --uri tcp://0.0.0.0:10200

# CPU mode
./script/run --uri tcp://0.0.0.0:10200 --cpu
```

### Systemd Deployment (LXC & Linux)
To install as a background service that starts on boot:

```bash
chmod +x script/install-service
./script/install-service
```
