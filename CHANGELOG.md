# Changelog

## Unreleased

- Sync changes from `mtk_qwen3-vl/llm_cmdline_tool-v1.9.2/` for Qwen3-VL bring-up:
  - New build scripts: `build_all_with_mllm.sh`, `build_clean.sh`, `build_libcommon.sh`, `build_libmllm.sh`, `build_libtokenizer.sh`, `build_mllm_runner.sh`, `build_runner.sh`.
  - New runner entry: `jni/main/main_mmbench.cpp`.
  - New prebuilt headers under `jni/prebuilt/include/` (`medusa_config.h`, `mtk_llm.h`, `mtk_llm_options.h`, `mtk_llm_types.h`, `mtk_mllm.h`, `tokenizer/`).
  - New prebuilt shared libs under `jni/prebuilt/` (`libcommon.so`, `libmtk_llm.so`, `libmtk_mllm.so`, `libopencv_java4.so`, `libtokenizer.so`).
  - Updates to `jni/main/Android_mllm.mk`, `jni/main/main.cpp`, `jni/main/main_llava.cpp`, `jni/prebuilt/src/image_utils/image_transform.{cpp,h}`, `jni/prebuilt/src/llm_runtime/{llava_mllm_runtime.cpp,mllm_runtime.cpp}`, `jni/prebuilt/src/mtk_llm_options.h`, `jni/utils/config_parser.cpp`.
  - Drop unused `jni/prebuilt/src/llm_runtime/llava_mllm_runtime.h`, `llava_in_the_wild_images.txt`, `llava_in_the_wild_questions.txt`.
