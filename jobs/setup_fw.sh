#!/bin/bash
set -e
R=${PBS_O_WORKDIR:-$PWD}   # repo root (submit jobs from the repo root)
export PIP_CACHE_DIR=$R/.cache/pip HF_HOME=$R/.cache/hf HF_HUB_DISABLE_XET=1
$R/env/bin/pip install -q faster-whisper "nvidia-cublas-cu12" "nvidia-cudnn-cu12==9.*"
$R/env/bin/python -c "import faster_whisper, ctranslate2; print('faster_whisper', faster_whisper.__version__, 'ct2', ctranslate2.__version__)"
$R/env/bin/python $R/jobs/dl_model.py Systran/faster-whisper-large-v3
echo FW_DONE
