#pragma once

namespace mtk::llm_runtime {

#ifdef USE_USDK_BACKEND
static constexpr bool kUseUsdkBackend = true;
#else
static constexpr bool kUseUsdkBackend = false;
#endif

#ifdef DISABLE_MULTITHREAD_MODEL_LOAD
static constexpr bool kUseMultiThreadedLoad = false;
#else
static constexpr bool kUseMultiThreadedLoad = true;
#endif

#ifdef DISABLE_INFERENCE_PIPELINING
static constexpr bool kUseInferencePipelining = false;
#else
static constexpr bool kUseInferencePipelining = true;
#endif

// Default enable left padding for LLM
#ifdef DISABLE_LLM_LEFT_PADDING
static constexpr bool kAllowLlmLeftPadding = false;
#else
static constexpr bool kAllowLlmLeftPadding = true;
#endif

// Default disable left padding for MLLM
#ifdef ALLOW_MLLM_LEFT_PADDING
static constexpr bool kAllowMllmLeftPadding = true;
#else
static constexpr bool kAllowMllmLeftPadding = false;
#endif

} // namespace mtk::llm_runtime