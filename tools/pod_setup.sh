#!/usr/bin/env bash
set -e
cd /workspace/sparseflow
export PYTHONPATH="$(pwd):$PYTHONPATH"
python -m pip install -U pip
python -m pip install -U "transformers==4.43.3" accelerate sentencepiece
python -c "import transformers, accelerate, torch; print('OK', transformers.__version__, accelerate.__version__, torch.__version__)"
