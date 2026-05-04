# Changelog

## Unreleased

- Sync changes from `mtk_qwen3-vl/llm_cmdline_tool-v1.9.2/` for Qwen3-VL bring-up:
  - New build scripts: `build_all_with_mllm.sh`, `build_clean.sh`, `build_libcommon.sh`, `build_libmllm.sh`, `build_libtokenizer.sh`, `build_mllm_runner.sh`, `build_runner.sh`.
  - New runner entry: `jni/main/main_mmbench.cpp`.
  - New prebuilt headers under `jni/prebuilt/include/` (`medusa_config.h`, `mtk_llm.h`, `mtk_llm_options.h`, `mtk_llm_types.h`, `mtk_mllm.h`, `tokenizer/`).
  - New prebuilt shared libs under `jni/prebuilt/` (`libcommon.so`, `libmtk_llm.so`, `libmtk_mllm.so`, `libopencv_java4.so`, `libtokenizer.so`).
  - Updates to `jni/main/Android_mllm.mk`, `jni/main/main.cpp`, `jni/main/main_llava.cpp`, `jni/prebuilt/src/image_utils/image_transform.{cpp,h}`, `jni/prebuilt/src/llm_runtime/{llava_mllm_runtime.cpp,mllm_runtime.cpp}`, `jni/prebuilt/src/mtk_llm_options.h`, `jni/utils/config_parser.cpp`.
- Restore `jni/prebuilt/src/llm_runtime/llava_mllm_runtime.h` and `llava_in_the_wild_{images,questions}.txt` that were mistakenly removed by an earlier `rsync --delete` sync. The header is still `#include`d by `mtk_mllm.cpp` and `llava_mllm_runtime.cpp` — the embedded copy under `mtk_qwen3-vl/` was simply missing it.
