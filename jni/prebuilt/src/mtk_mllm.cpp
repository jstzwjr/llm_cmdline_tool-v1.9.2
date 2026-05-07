#include "mtk_mllm.h"

#include "common/logging.h"
#include "executor/shared_weights.h"
#include "llm_runtime/llava_mllm_runtime.h"
#include "tokenizer/tokenizer.h"

#include <algorithm>
#include <regex>
#include <string>
#include <string_view>
#include <thread>

#define LLM_API __attribute__((visibility("default")))

using TokenType = mtk::Tokenizer::TokenType;
using mtk::LogitsKind;

bool LLM_API mtk_mllm_init(void** runtime, const LlmModelOptions& modelOptions,
                           const LlavaRuntimeOptions& runtimeOptions,
                           const SharedWeightsHandle* preloadedSharedWeights) {
    auto llavaRuntime = std::make_unique<mtk::LlavaRuntime>(kImagePlaceholderToken);
    auto preloadSw = reinterpret_cast<const mtk::SharedWeightsHandle*>(preloadedSharedWeights);
    const auto status = llavaRuntime->initialize(modelOptions, runtimeOptions, preloadSw);
    *runtime = llavaRuntime.release();
    return status;
}

void* LLM_API mtk_mllm_inference_once(void* runtime, const size_t leftPadSize,
                                      const size_t rightPadSize, const void* inputEmb,
                                      const LogitsKind outputKind) {
    // NOTE: Mllm default used right padding
    auto llavaRuntime = reinterpret_cast<mtk::LlavaRuntime*>(runtime);
    return llavaRuntime->run(inputEmb, leftPadSize, rightPadSize, outputKind);
}

void LLM_API mtk_mllm_release(void* runtime) {
    auto llavaRuntime = reinterpret_cast<mtk::LlavaRuntime*>(runtime);
    llavaRuntime->release();
    delete llavaRuntime;
}

size_t LLM_API mtk_mllm_get_per_token_logits_size(void* runtime) {
    auto llavaRuntime = reinterpret_cast<mtk::LlavaRuntime*>(runtime);
    return llavaRuntime->getPerTokenLogitsSize();
}

double LLM_API mtk_mllm_get_last_clip_elapsed_seconds(void* runtime) {
    auto llavaRuntime = reinterpret_cast<mtk::LlavaRuntime*>(runtime);
    return llavaRuntime->getLastClipElapsedSeconds();
}

void* LLM_API mtk_mllm_consume_prompt(void* runtime, const std::vector<TokenType>& tokens,
                                      const std::vector<std::string>& imagePaths,
                                      size_t* numPromptToken, const LogitsKind outputKind) {
    auto llavaRuntime = reinterpret_cast<mtk::LlavaRuntime*>(runtime);

    // Load the images for the current prompt
    std::vector<FileMemMapper> fileMemMapperPool;
    std::vector<std::string_view> imageBuffers;
    for (const auto& imagePath : imagePaths) {
        FileMemMapper imageFile(imagePath);
        const auto [buffer, size] = imageFile.get();
        imageBuffers.emplace_back(buffer, size);
        fileMemMapperPool.emplace_back(std::move(imageFile));
    }
    return llavaRuntime->consumePrompt(tokens, imageBuffers, numPromptToken, outputKind);
}

void* LLM_API mtk_mllm_consume_emb(void* runtime, const char* embBuffer, const size_t embBufferSize,
                                   const LogitsKind outputKind) {
    auto llavaRuntime = reinterpret_cast<mtk::LlavaRuntime*>(runtime);
    return llavaRuntime->consumeEmbeddings(embBuffer, embBufferSize, outputKind);
}

size_t LLM_API mtk_mllm_get_token_index(void* runtime) {
    auto llavaRuntime = reinterpret_cast<mtk::LlavaRuntime*>(runtime);
    return llavaRuntime->getTokenIndex();
}

void LLM_API mtk_mllm_rollback(void* runtime, const size_t rollbackCount) {
    auto llavaRuntime = reinterpret_cast<mtk::LlavaRuntime*>(runtime);
    llavaRuntime->rollback(rollbackCount);
}

void* LLM_API mtk_mllm_get_text_embedding(void* runtime,
                                          const std::vector<TokenType>& inputTokens) {
    auto llavaRuntime = reinterpret_cast<mtk::LlavaRuntime*>(runtime);
    return llavaRuntime->getTextEmbedding(inputTokens);
}

void* LLM_API mtk_mllm_get_clip_embedding(void* runtime, void* imageBuffer,
                                          const size_t imageBufferSize) {
    auto llavaRuntime = reinterpret_cast<mtk::LlavaRuntime*>(runtime);
    return llavaRuntime->getImageEmbedding(imageBuffer, imageBufferSize);
}

void LLM_API mtk_mllm_reset(void* runtime, const bool resetCache) {
    auto llavaRuntime = reinterpret_cast<mtk::LlavaRuntime*>(runtime);
    llavaRuntime->reset(resetCache);
}