#pragma once

#include "llm_runtime/mllm_runtime.h"

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

    UNUSED_MODALITY(Audio)

private:
    // Hide to prevent it from being accidentally used
    using LlmRuntime::initialize;

private:
    std::unique_ptr<Executor> mClipPatchEmbExecutor;
    std::unique_ptr<Executor> mClipExecutor;
};

} // namespace mtk