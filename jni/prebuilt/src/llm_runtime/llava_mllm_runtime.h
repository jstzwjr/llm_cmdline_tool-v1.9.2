#pragma once

#include "llm_runtime/mllm_runtime.h"
#include "mtk_llm_options.h"

#include <string>
#include <vector>

#define LLM_API __attribute__((visibility("default")))

namespace mtk {

// Multi-modal LLM Runtime
class LLM_API LlavaRuntime : public MllmRuntime {
public:
    LlavaRuntime();

    virtual ~LlavaRuntime();

    explicit LlavaRuntime(const Tokenizer::TokenType imagePlaceholderToken);

    bool initialize(const LlmModelOptions& modelOptions, const LlavaRuntimeOptions& runtimeOptions,
                    const SharedWeightsHandle* preloadedSharedWeights = nullptr);

    virtual void release() override;

    virtual void* getImageEmbedding(const void* buffer, const size_t size) override;

    virtual size_t getImageEmbeddingSize() const override;

    // Phase 3.1 (deepstack): get an extra embedding buffer cached from the last
    // getImageEmbedding() call. idx is 0-based; size is set by deepstack tap count
    // detected at runtime (Qwen3-VL = 3, others = 0).
    void* getDeepstackEmbedding(const size_t idx) const override;
    size_t getDeepstackEmbeddingSize(const size_t idx) const override;
    size_t getNumDeepstackEmbeddings() const override { return mDeepstackEmbeddings.size(); }

    UNUSED_MODALITY(Audio)

private:
    // Hide to prevent it from being accidentally used
    using LlmRuntime::initialize;

private:
    std::unique_ptr<Executor> mClipPatchEmbExecutor;
    std::unique_ptr<Executor> mClipExecutor;

    // Pre-existing field used by getImageEmbedding (declaration was lost in an earlier
    // mtk_qwen3-vl sync; restoring it here so the build compiles).
    ClipPreprocessOptions mClipPreprocess;

    // Phase 3.1 (deepstack):
    // mDeepstackOutputIndices[k] = which CLIP DLA output port carries the k-th
    //                              deepstack embedding (set in initialize).
    // mDeepstackEmbeddings[k] / mDeepstackEmbeddingSizes[k] cache pointer + bytes
    //                              from the most recent CLIP forward pass.
    std::vector<size_t> mDeepstackOutputIndices;
    std::vector<void*> mDeepstackEmbeddings;
    std::vector<size_t> mDeepstackEmbeddingSizes;
};

} // namespace mtk