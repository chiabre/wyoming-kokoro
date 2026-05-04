# 🎙️ Wyoming Kokoro TTS (GPU/CPU)

[Wyoming Protocol](https://github.com/rhasspy/wyoming) server for high-speed local Text-to-Speech using [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), optimized for Home Assistant voice pipelines.

---

## ⚡ Hardware Selection

This repository is hardware-aware and automatically selects the best execution mode based on available system resources.

| Feature | 🟢 GPU Mode (Recommended) | 🔵 CPU Mode |
| :--- | :--- | :--- |
| **Performance** | Near real-time (fast synthesis) | Slower but stable |
| **Requirements** | NVIDIA GPU (CUDA / onnxruntime-gpu) | Any modern CPU |
| **Model** | FP16 | FP32 / INT8 |

---

## ⚙️ Installation

```bash
git clone https://github.com/chiabre/wyoming-kokoro.git
cd wyoming-kokoro
chmod +x script/setup
./script/setup
```

> [!IMPORTANT]
> **GPU Users:** reload your environment after setup:
> 
> ```bash
> source ~/.bashrc
> ```

---

## 🧠 Model Selection

Kokoro supports multiple ONNX model formats depending on hardware.

You can select models via the installer or manually using --model.

### ⚙️ Supported Models

| Alias     | Model                   | Best For         | Notes                           |
| --------- | ----------------------- | ---------------- | ------------------------------- |
| 🏆 `fp16` | `kokoro-v1.0.fp16.onnx` | NVIDIA GPU       | Best balance of speed & quality |
| 💻 `fp32` | `kokoro-v1.0.onnx`      | CPU systems      | Most compatible                 |
| 🪶 `int8` | `kokoro-v1.0.int8.onnx` | Edge / low power | Smallest footprint              |


### 📥 Model Setup

Model and voice files are handled automatically by the setup script.

The installer will:
- Detect available GPU/CPU hardware
- Prompt for model selection (FP16 / FP32 / INT8)
- Download the correct files into ./data
- Ensure version compatibility (Kokoro v1.0 + voices v1.0

#### ⚙️ Run the automated setup

```bash
./script/model
```
#### 🧠 Model selection (during setup)
You will be prompted to choose:

| Option | Description                                      |
| ------ | ------------------------------------------------ |
| FP16   | 🟢 Recommended for NVIDIA GPU (best performance) |
| FP32   | 🔵 CPU compatible (stable fallback)              |
| INT8   | 🪶 Low-power / edge devices                      |


---

## 🚀 Running the Service

### Basic Usage

```bash
python3 -m wyoming_kokoro
```
### Examples

```bash
# GPU mode (recommended)
python3 -m wyoming_kokoro --model data/kokoro-v1.0.fp16.onnx

# CPU mode
python3 -m wyoming_kokoro --model data/kokoro-v1.0.onnx --cpu

# Edge mode
python3 -m wyoming_kokoro --model data/kokoro-v1.0.int8.onnx --cpu

# Debug mode
python3 -m wyoming_kokoro --debug

# Custom port
python3 -m wyoming_kokoro --uri tcp://0.0.0.0:10200
```

---

## ⚙️ Configuration Options
| Parameter    | Type     | Default               | Description             |
| :----------- | :------- | :-------------------- | :---------------------- |
| `--uri`      | Optional | `tcp://0.0.0.0:10200` | Wyoming server address  |
| `--data-dir` | Optional | `./data`              | Model directory         |
| `--model`    | Optional | auto                  | Path to ONNX model      |
| `--voices`   | Optional | auto                  | Voices bin file         |
| `--voice`    | Optional | `af_heart`            | Default voice           |
| `--speed`    | Optional | `1.0`                 | Speech speed multiplier |
| `--cpu`      | Flag     | `False`               | Force CPU inference     |
| `--debug`    | Flag     | `False`               | Enable verbose logs     |

---

## 🧩 Systemd Deployment
Install as a background service:

```bash
chmod +x script/install-service
./script/install-service
```

---

## 🔧 Service Management

- Logs:

```bash
journalctl -u wyoming-kokoro -f
```

- Restart:
  
```bash
systemctl restart wyoming-kokoro
```
---

## 🧠 Notes

- Use `FP16` for best GPU performance 
- Use `FP32` for maximum compatibility
- Use `INT8` for embedded or low-power systems
- Model selection is fully manual or installer-driven
- Voices must always match version v1.0
