# Local LLM

A local inference prototype for the Llama 3.3 8B Instruct GGUF model. It
provides an OpenAI-compatible API and a small Gradio chat UI.

> **This project currently runs inference; it does not train or fine-tune a
> model.** No user training dataset or training pipeline is present. The
> checked-in model directory is local runtime data, not part of the Docker
> image or source distribution.

## Requirements

- Python 3.12 for the local setup, or Docker Engine with the Compose plugin.
- The GGUF model at
  `models/allura-forge_Llama-3.3-8B-Instruct-Q4_K_M.gguf` (about 4.6 GiB).
- For CPU inference, allow several additional GiB of RAM for the model,
  context cache, and application. More context and concurrent requests need
  more memory.

The model is a quantized inference artifact. Review the model's [license and
terms](https://huggingface.co/bartowski/allura-forge_Llama-3.3-8B-Instruct-GGUF)
before using or distributing it.

## Run locally

From the project root, create and activate an environment, then install the
pinned direct dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`llama-cpp-python` builds a CPU backend by default. Building it from source can
take several minutes and requires a C/C++ compiler and CMake. To use optimized
OpenBLAS on Linux, install its development headers and set the build options
before installing:

```bash
sudo apt-get install build-essential cmake libopenblas-dev
export CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
python -m pip install -r requirements.txt
```

Start the API and UI in separate terminals:

```bash
# Terminal 1: OpenAI-compatible API on http://localhost:8000
python server.py

# Terminal 2: browser UI on http://localhost:7860
python ui.py
```

The API loads the model during startup, so its first response is not available
until loading completes. A basic API check is:

```bash
curl http://localhost:8000/v1/models
```

The one-shot inference smoke test loads the same model and prints a response:

```bash
python run.py
```

### Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_PATH` | `models/allura-forge_Llama-3.3-8B-Instruct-Q4_K_M.gguf` | GGUF model file |
| `MODEL_ALIAS` / `LLM_MODEL` | `local-llm` | API model name |
| `N_CTX` | `2048` | Context length |
| `N_GPU_LAYERS` | `0` | Layers offloaded to a CUDA-enabled llama.cpp build |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | API listen address and port |
| `LLM_API_URL` | `http://localhost:8000/v1/chat/completions` | API URL used by the UI |
| `UI_HOST` / `UI_PORT` | `0.0.0.0` / `7860` | UI listen address and port |
| `SYSTEM_PROMPT` | A short local-assistant instruction | System message used by the UI |

The default dependency installation is CPU-only. Setting `N_GPU_LAYERS` alone
does not enable GPU support; use a CUDA-enabled llama.cpp build and a compatible
NVIDIA container runtime before offloading layers.

## Docker deployment

The Compose setup builds a CPU/OpenBLAS image and mounts `./models` read-only,
so the multi-gigabyte model is not copied into the image. From the project
root, with the GGUF in the documented path:

```bash
docker compose up --build
```

Open <http://localhost:7860>. The API is also available at
<http://localhost:8000/v1/models>. Both host ports are bound to loopback, not
all network interfaces. The UI starts after the API model is loaded and
healthy. Startup can take several minutes on CPU.

Useful operations:

```bash
docker compose logs -f api ui
docker compose ps
docker compose down
```

If the Docker Compose plugin is unavailable, run the same image directly:

```bash
docker build -t local-llm:dev .
docker network create local-llm
docker run -d --name local-llm-api --network local-llm \
  --restart unless-stopped \
  -p 127.0.0.1:8000:8000 \
  -v "$PWD/models:/models:ro" \
  -e MODEL_PATH=/models/allura-forge_Llama-3.3-8B-Instruct-Q4_K_M.gguf \
  -e MODEL_ALIAS=local-llm -e N_CTX=2048 -e N_GPU_LAYERS=0 \
  -e HOST=0.0.0.0 -e PORT=8000 \
  local-llm:dev
docker run -d --name local-llm-ui --network local-llm \
  --restart unless-stopped \
  -p 127.0.0.1:7860:7860 \
  -e LLM_API_URL=http://local-llm-api:8000/v1/chat/completions \
  -e LLM_MODEL=local-llm \
  local-llm:dev python ui.py
```

Stop and remove the manually launched containers and network with:

```bash
docker rm -f local-llm-ui local-llm-api
docker network rm local-llm
```

To use a different GGUF path or context size, edit the `MODEL_PATH` or `N_CTX`
environment entries in `compose.yaml`; ensure the host model is under the
mounted `models/` directory. Rebuild after changing dependencies or the
Dockerfile. The model itself remains on the host when Compose is stopped.

The supplied image does not include CUDA. A GPU deployment needs an NVIDIA
Container Toolkit-compatible host, a CUDA-enabled `llama-cpp-python` build,
GPU device passthrough in Compose, and a tested `N_GPU_LAYERS` setting sized to
available VRAM. Treat that as a separate image/build configuration; don't just
change the layer count in the CPU image.

Do not publish the API or unauthenticated UI directly to the public internet.
For remote access, add authentication and TLS at a trusted reverse proxy or use
a private VPN, and bind only the proxy's required ports. The UI disables
Gradio analytics by default; model requests are handled by the local API.

## Development strategy

1. **Make inference reproducible first.** Keep Python and model versions
   explicit, store model weights outside the image, and smoke-test the API and
   chat format whenever dependencies or model artifacts change.
2. **Define the data task before training.** Decide whether the goal is to
   retrieve private or frequently changing facts, improve response style, or
   teach a repeatable task. For retrieval and frequently changing knowledge,
   start with retrieval-augmented generation (RAG) rather than changing model
   weights. For stable behavior or task-specific responses, evaluate a
   parameter-efficient fine-tune such as LoRA/QLoRA.
3. **Prepare a lawful, high-quality dataset.** Record provenance and usage
   rights, remove secrets and personal data unless there is a documented basis
   to use them, deduplicate and validate examples, and create separate
   train/validation/test splits. Keep test examples out of training.
4. **Fine-tune the source model, not this GGUF quantization.** Use a suitable
   training checkpoint and a dedicated training workflow (for example,
   Transformers/PEFT); the current GGUF is quantized for inference. Evaluate
   the adapter against the untouched baseline, then merge/convert and quantize
   an approved result to GGUF for llama.cpp deployment.
5. **Treat training and serving as separate workloads.** Version the data
   preparation, training configuration, evaluation results, adapter, and
   converted model. Don't put datasets or checkpoints in the serving image.
   A 4 GB laptop GPU is unlikely to be a practical target for fine-tuning an
   8B model; investigate quantized adapter training on suitable hardware and
   measure actual memory needs before committing to that plan.
6. **Add operational controls before sharing it.** Add tests for API/chat
   behavior, model and data provenance, resource limits, authentication if
   exposed beyond loopback, and a repeatable release/deployment process.

## Troubleshooting

- **Model file not found:** check the exact `MODEL_PATH`; in Compose verify the
  host file exists under `models/` and is readable.
- **Slow startup or out-of-memory:** start with `N_CTX=2048`, use CPU as the
  configured default, close competing applications, and confirm enough RAM
  for the weights plus runtime cache.
- **UI cannot reach the API:** start `python server.py` first for local use;
  under Compose, inspect `docker compose logs api` and wait for its health
  check to pass.
- **Unexpected model responses:** use a GGUF compatible with llama.cpp and its
  embedded chat template. If needed, set a supported `CHAT_FORMAT` for the
  specific model.
