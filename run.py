import os
from pathlib import Path

from llama_cpp import Llama


model_path = Path(
    os.getenv(
        "MODEL_PATH",
        str(
            Path(__file__).resolve().parent
            / "models"
            / "allura-forge_Llama-3.3-8B-Instruct-Q4_K_M.gguf"
        ),
    )
).expanduser()
if not model_path.is_file():
    raise FileNotFoundError(
        f"Model file not found: {model_path}. Set MODEL_PATH to a readable GGUF file."
    )

llm = Llama(
    model_path=str(model_path),
    n_ctx=int(os.getenv("N_CTX", "2048")),
    n_gpu_layers=int(os.getenv("N_GPU_LAYERS", "0")),
    verbose=False,
)

response = llm.create_chat_completion(
    messages=[{"role": "user", "content": "Hello! Briefly introduce yourself."}],
    max_tokens=100,
    temperature=0.7,
)
print(response["choices"][0]["message"]["content"])
