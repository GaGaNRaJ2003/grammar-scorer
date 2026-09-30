#!/bin/bash
# Private env for this project only; caches kept inside the project dir so nothing outside it is touched.
set -e
R=${PBS_O_WORKDIR:-$PWD}   # repo root (submit jobs from the repo root)
export PIP_CACHE_DIR=$R/.cache/pip CONDA_PKGS_DIRS=$R/.cache/conda
conda create -y -q -p $R/env python=3.11
$R/env/bin/pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu121
$R/env/bin/pip install -q "transformers>=4.44" accelerate soundfile librosa pandas scikit-learn lightgbm scipy matplotlib jupyter
$R/env/bin/python -c "import torch, transformers; print('torch', torch.__version__, 'cuda build', torch.version.cuda, 'transformers', transformers.__version__)"
echo ENV_DONE
