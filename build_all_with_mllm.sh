#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

bash build_clean.sh

bash build_libcommon.sh
bash build_libtokenizer.sh
bash build_libmllm.sh
bash build_runner.sh
bash build_mllm_runner.sh
