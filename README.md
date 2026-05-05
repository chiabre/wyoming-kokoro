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

### ⚙️ Supported Models

| Alias     | Model                   | Best For         | Notes                           |
| --------- | ----------------------- | ---------------- | ------------------------------- |
| 🏆 `fp16` | `kokoro-v1.0.fp16.onnx` | NVIDIA GPU       | Best balance of speed & quality |
| 💻 `fp32` | `kokoro-v1.0.onnx`      | CPU systems      | Most compatible                 |
| 🪶 `int8` | `kokoro-v1.0.int8.onnx` | Edge / low power | Smallest footprint              |


### 📥 Model Setup

Model and voice files are handled automatically by the `./script/model` script.

The script  will:
- Detect available GPU/CPU hardware
- Prompt for model selection (FP16 / FP32 / INT8)
- Download the correct files into `./data`

#### ⚙️ Run the automated setup

```bash
./script/model
```

## 📥 Model Resolution

The runtime **does NOT hardcode a model path**.

Instead it uses this logic:

```text
--model (CLI override)
↓
./data/*.onnx (auto-pick first match)
↓
./data/kokoro-v1.0.onnx (fallback)
```

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
| `--voice`    | Optional | `af_heart`            | Default fallback voice  |
| `--speed`    | Optional | `1.0`                 | Speech speed multiplier |
| `--cpu`      | Flag     | `False`               | Force CPU inference     |
| `--debug`    | Flag     | `False`               | Enable verbose logs     |

---

## 🧠 Voice Behavior

### 🎤 Voice selection priority

1. Client-provided voice (Wyoming / Home Assistant)
2. CLI `--voice`
3. Default fallback: `af_heart`

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
- Model selection is installer-driven or manual via CLI
- Voice is now client-overridable (Wyoming compliant)
- Warmup is performed at startup to reduce first-request latency
