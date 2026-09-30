"""Convert an official HF checkpoint's pytorch_model.bin to model.safetensors in models/<name>/.
transformers 5 refuses .bin files with torch < 2.6; these are official Microsoft weights, loaded tensors-only."""
import shutil, sys
from pathlib import Path
import torch
from safetensors.torch import save_file
import os

repo = sys.argv[1]
src = next((Path(os.environ["HF_HOME"]) / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots").iterdir())
dst = Path("models") / repo.split("/")[-1]
dst.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    if f.name != "pytorch_model.bin":
        shutil.copy(f.resolve(), dst / f.name)
sd = torch.load(src / "pytorch_model.bin", map_location="cpu", weights_only=True)
save_file({k: v.contiguous().clone() for k, v in sd.items()}, dst / "model.safetensors", metadata={"format": "pt"})
print("CONVERTED", repo, "->", dst, len(sd), "tensors")
