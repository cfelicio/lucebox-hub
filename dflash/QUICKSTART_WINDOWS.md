# DFlash Windows Quickstart

## What is this?

DFlash is **not** llama.cpp. It's a standalone C++/CUDA speculative decoding engine built directly on ggml. There is no `llama-server.exe` — instead:

| llama.cpp equivalent | DFlash equivalent |
|---|---|
| `llama-server.exe` | `python scripts/server.py` (OpenAI-compatible HTTP server, uses `test_dflash.exe` internally) |
| `llama-cli.exe` | `python scripts/run.py` (one-shot streaming generation) |
| N/A | `python examples/chat.py` (multi-turn chat REPL) |
| `test_dflash.exe` is the core engine, not meant to be called directly | |

DFlash runs Qwen3.5-27B at **129.5 tok/s** (vs 37.7 tok/s autoregressive) via block-diffusion speculative decoding + DDTree verify.

---

## Directory layout (after unzipping release)

```
dflash-windows/
├── bin/
│   ├── test_dflash.exe          # Core DFlash speculative decoder
│   ├── test_generate.exe        # Autoregressive-only decoder (baseline)
│   ├── test_kv_quant.exe        # KV quant unit tests
│   ├── ggml-base.dll            # ggml runtime
│   ├── ggml-cuda.dll            # CUDA backend
│   ├── ggml-cpu.dll             # CPU backend
│   └── ggml.dll                 # Backend loader
├── scripts/
│   ├── run.py                   # One-shot streaming generation
│   ├── server.py                # OpenAI-compatible HTTP server
│   ├── server_tools.py          # Server with tool-calling support
│   ├── bench_llm.py             # Full benchmark suite
│   └── ...
├── examples/
│   └── chat.py                  # Multi-turn chat REPL
├── README.md
└── RESULTS.md
```

---

## Step 0: Prerequisites

You need:
- **Python 3.10+** ([python.org/downloads](https://www.python.org/downloads/))
- **NVIDIA GPU** with CUDA drivers installed (`nvidia-smi` should work)

### Set up a virtual environment

```powershell
# Create venv (run from the dflash-windows folder)
python -m venv .venv

# Activate it
.venv\Scripts\Activate.ps1

# Install all Python dependencies
pip install transformers huggingface-hub fastapi uvicorn jinja2
```

> **Note:** Every time you open a new terminal, activate the venv first:
> ```powershell
> .venv\Scripts\Activate.ps1
> ```

---

## Step 1: Download models (~20 GB)

```powershell
# Target model: Qwen3.6-27B Q4_K_M (~16 GB)
hf download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q4_K_M.gguf --local-dir models/

# Draft model: z-lab DFlash BF16 (~3.5 GB)
hf download z-lab/Qwen3.6-27B-DFlash model.safetensors --local-dir models/draft/
```

After downloading, your directory should look like:
```
dflash-windows/
├── bin/
├── scripts/
├── models/
│   ├── Qwen3.6-27B-Q4_K_M.gguf
│   └── draft/
│       └── model.safetensors
└── ...
```

---

## Step 2: Choose your usage mode

### Option A: One-shot generation (equivalent to `llama-cli`)

```powershell
python scripts/run.py --prompt "def fibonacci(n):" --bin bin/test_dflash.exe --target models/Qwen3.6-27B-Q4_K_M.gguf
```

With system prompt:
```powershell
python scripts/run.py --prompt "Explain quantum computing" --system "You are a concise physics tutor." --bin bin/test_dflash.exe --target models/Qwen3.6-27B-Q4_K_M.gguf
```

Generate more tokens:
```powershell
python scripts/run.py --prompt "Write a short story about AI" --n-gen 512 --bin bin/test_dflash.exe --target models/Qwen3.6-27B-Q4_K_M.gguf
```

### Option B: OpenAI-compatible HTTP server (equivalent to `llama-server`)

This is the direct equivalent of your llama-server command:

```powershell
# Your old command:
#   .\llama-server.exe --model Qwen3.6-27B-UD-Q4_K_XL.gguf --temp 0.6 --cache-type-k turbo4 --cache-type-v turbo4
#
# DFlash equivalent:
python scripts/server.py --bin bin/test_dflash.exe --target models/Qwen3.6-27B-Q4_K_M.gguf --port 8080
```

The server is now running at `http://localhost:8080`. Use it with:

```powershell
# curl test
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"luce-dflash\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"stream\":true}"
```

**Connect Open WebUI / LM Studio / Cline:**
```
OPENAI_API_BASE=http://localhost:8080/v1
OPENAI_API_KEY=sk-any
```

### Option C: Interactive chat REPL

```powershell
python examples/chat.py
```

Type your messages, get streaming responses. Ctrl+C to cancel a reply, Ctrl+D to exit.

---

## Step 3: Tuning options

### KV cache quantization (equivalent to `--cache-type-k turbo4`)

DFlash uses TurboQuant (TQ3_0) instead of llama.cpp's "turbo4":

```powershell
# TQ3_0 KV cache (3.5 bits per value, fits 256K context in 24 GB)
python scripts/server.py --bin bin/test_dflash.exe --ctk tq3_0 --ctv tq3_0

# Q4_0 KV cache (fits up to ~128K context)
python scripts/server.py --bin bin/test_dflash.exe --ctk q4_0 --ctv q4_0

# Q8_0 KV cache (default, best quality, shorter context)
python scripts/server.py --bin bin/test_dflash.exe --ctk q8_0 --ctv q8_0

# Asymmetric: Q8_0 keys + Q4_0 values
python scripts/server.py --bin bin/test_dflash.exe --ctk q8_0 --ctv q4_0

# F16 KV (no quantization, shortest context)
python scripts/server.py --bin bin/test_dflash.exe --kv-f16
```

### Context length

```powershell
# Default: auto-fit to prompt+generation (recommended)
python scripts/server.py --bin bin/test_dflash.exe

# Long context server (auto-enables TQ3_0 KV when >6144)
python scripts/server.py --bin bin/test_dflash.exe --max-ctx 32768

# 128K context (requires TQ3_0 KV, explicitly set)
python scripts/server.py --bin bin/test_dflash.exe --max-ctx 131072 --ctk tq3_0 --ctv tq3_0
```

> **WARNING**: Don't set `--max-ctx` much larger than you need. Attention compute scales with the allocated max_ctx, not the actual prompt length. A `--max-ctx=131072` on a 4K prompt runs attention 30× slower than necessary.

### DDTree budget (speculative decoding tree size)

```powershell
# Default budget=22 (sweet spot for RTX 3090 + Q4_K_M)
python scripts/run.py --prompt "hello" --budget 22 --bin bin/test_dflash.exe

# Smaller budget (faster per-step, lower acceptance)
python scripts/run.py --prompt "hello" --budget 16 --bin bin/test_dflash.exe
```

### Sliding-window attention

```powershell
# Default: window=2048 (kicks in at long contexts, lossless for this model)
python scripts/server.py --bin bin/test_dflash.exe --fa-window 2048

# Full attention (slower at long context, but no windowing)
python scripts/server.py --bin bin/test_dflash.exe --fa-window 0
```

---

## Qwen3.6-27B (your current model, supported!)

If you're currently using Qwen3.6-27B with llama.cpp, DFlash supports it as a drop-in target.

### Full setup (copy-paste into PowerShell)

```powershell
# ── 1. Activate venv ──
.venv\Scripts\Activate.ps1

# ── 2. Download models ──
# Target: Qwen3.6-27B Q4_K_M (~16 GB)
hf download unsloth/Qwen3.6-27B-GGUF Qwen3.6-27B-Q4_K_M.gguf --local-dir models/

# Draft: matched DFlash draft (~3.5 GB, still training but works)
hf download z-lab/Qwen3.6-27B-DFlash model.safetensors --local-dir models/draft/

# ── 3. Run the server (max performance for RTX 3090) ──
python scripts/server.py `
  --bin bin/test_dflash.exe `
  --target models/Qwen3.6-27B-Q4_K_M.gguf `
  --draft models/draft `
  --budget 22 `
  --ctk tq3_0 --ctv tq3_0 `
  --max-ctx 16384 `
  --port 8080
```

The server is now running at `http://localhost:8080`. Connect Open WebUI, Cline, or any OpenAI client:
```
API Base: http://localhost:8080/v1
API Key:  sk-any
Model:    luce-dflash
```

### One-shot generation

```powershell
python scripts/run.py `
  --prompt "Write a Python function that sorts a list" `
  --target models/Qwen3.6-27B-Q4_K_M.gguf `
  --bin bin/test_dflash.exe `
  --n-gen 512
```

### Max performance flags explained

| Flag | What it does | Why |
|---|---|---|
| `--budget 22` | DDTree verify with 22-node tree | Sweet spot for RTX 3090 + Q4_K_M (diminishing returns beyond 22) |
| `--ctk tq3_0 --ctv tq3_0` | TurboQuant 3.5 bpv KV cache | Fits longer context in 24 GB, near-lossless quality |
| `--max-ctx 16384` | 16K context window | Fits most API workloads; don't oversize (attention scales with this) |
| `--port 8080` | Server port | Match your client config |

> **Note:** Qwen3.6 throughput is ~70 tok/s vs 130 tok/s on 3.5 — the DFlash draft is still being trained for 3.6. Still **2× faster than autoregressive** (35 tok/s).

---

## Side-by-side: llama.cpp vs DFlash

| Feature | llama.cpp | DFlash |
|---|---|---|
| **Decode speed** (27B, RTX 3090) | ~38 tok/s (AR) | **~130 tok/s** (spec decode) |
| **Technique** | Autoregressive | Block-diffusion + DDTree |
| **Server** | `llama-server.exe` | `python scripts/server.py` |
| **API** | OpenAI compatible | OpenAI + Anthropic compatible |
| **KV cache quant** | `turbo4` | `tq3_0` (3.5 bpv) / `q4_0` / `q8_0` |
| **Max context** | Depends on VRAM | Up to 256K (24 GB, TQ3_0) |
| **Temperature/sampling** | Full support | Greedy only (temperature accepted but ignored) |
| **Batching** | Multi-user | Single-user (batch=1) |
| **Models** | Many architectures | Qwen3.5-27B / Qwen3.6-27B only |

### Key differences from llama.cpp

1. **No sampling** — DFlash is greedy-only. `--temp`, `--top-p`, `--top-k` are accepted by the server for API compatibility but silently ignored.
2. **Two models required** — DFlash needs both a target GGUF (~16 GB) and a draft safetensors (~3.5 GB).
3. **Single model family** — Only Qwen3.5/3.6-27B. Not a general-purpose runtime.
4. **3.4× faster** — The tradeoff is worth it if you're running Qwen 27B on a single GPU.

---

## Troubleshooting

### "DLL not found" errors
Make sure the `bin/` directory is in your PATH, or run from the `dflash-windows/` directory:
```powershell
$env:PATH = "$(Get-Location)\bin;$env:PATH"
```

### "CUDA error" at startup
- Verify NVIDIA drivers are installed: `nvidia-smi`
- DFlash requires CUDA-capable GPU (RTX 3090/4090)

### "target GGUF not found"
- Download models first (Step 1)
- Check the `models/` directory is next to `bin/` and `scripts/`

### Server hangs on first request
- First request loads both models (~10s). Subsequent requests are instant (daemon mode).
