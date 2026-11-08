import os
from pathlib import Path

import uvicorn
from llama_cpp.server.app import create_app
from llama_cpp.server.settings import ModelSettings, ServerSettings


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "allura-forge_Llama-3.3-8B-Instruct-Q4_K_M.gguf"
)
model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))).expanduser()
if not model_path.is_file():
    raise FileNotFoundError(
        f"Model file not found: {model_path}. Set MODEL_PATH to a readable GGUF file."
    )

app = create_app(
    model_settings=[
        ModelSettings(
            model=str(model_path),
            model_alias=os.getenv("MODEL_ALIAS", "lumin"),
            n_ctx=int(os.getenv("N_CTX", "2048")),
            n_gpu_layers=int(os.getenv("N_GPU_LAYERS", "0")),
            chat_format=os.getenv("CHAT_FORMAT") or None,
            verbose=False,
        )
    ],
    server_settings=ServerSettings(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    ),
)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )
