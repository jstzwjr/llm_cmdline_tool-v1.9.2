#include "mtk_llm.h"

#include "common/logging.h"
#include "executor/shared_weights.h"
#include "llm_helper/include/utils.h"
#include "llm_runtime/llm_runtime.h"
#include "tokenizer/tokenizer.h"

#include <memory>

#define LLM_API __attribute__((visibility("default")))

using TokenType = mtk::Tokenizer::TokenType;
using mtk::LogitsKind;

void LLM_API mtk_llm_shared_weights_create_handle(SharedWeightsHandle** sharedWeightsHandle,
                                                  const LlmRuntimeOptions& runtimeOptions) {
    auto swHandle = reinterpret_cast<mtk::SharedWeightsHandle**>(sharedWeightsHandle);
    *swHandle = mtk::createSharedWeightsHandle(runtimeOptions).release();
}

void LLM_API mtk_llm_shared_weights_release_handle(SharedWeightsHandle* sharedWeightsHandle) {
    auto swHandle = reinterpret_cast<mtk::SharedWeightsHandle*>(sharedWeightsHandle);
    mtk::freePreloadedSharedWeights(swHandle);
}

void LLM_API mtk_llm_shared_weights_preload(SharedWeightsHandle** sharedWeightsHandle,
                                            const LlmRuntimeOptions& runtimeOptions) {
    auto swHandle = reinterpret_cast<mtk::SharedWeightsHandle**>(sharedWeightsHandle);
    mtk::preloadSharedWeights(swHandle, runtimeOptions);
}

void LLM_API
mtk_llm_shared_weights_preload_union(const std::vector<SharedWeightsHandle*>& preloadHandles,
                                     const std::vector<SharedWeightsHandle*>& refHandles) {
    auto castVec = [](const auto& vec) {
        std::vector<mtk::SharedWeightsHandle*> result;
        result.reserve(vec.size());
        for (const auto& item : vec) {
            result.push_back(reinterpret_cast<mtk::SharedWeightsHandle*>(item));
        }
        return result;
    };
    mtk::preloadSharedWeightsUnion(castVec(preloadHandles), castVec(refHandles));
}

void LLM_API mtk_llm_shared_weights_unload(SharedWeightsHandle* sharedWeightsHandle,
                                           const size_t index) {
    auto swHandle = reinterpret_cast<mtk::SharedWeightsHandle*>(sharedWeightsHandle);
    mtk::unloadSharedWeights(swHandle, index);
}

bool LLM_API mtk_llm_init(void** runtime, const LlmModelOptions& modelOptions,
                          const LlmRuntimeOptions& runtimeOptions,
                          const SharedWeightsHandle* preloadedSharedWeights) {
    auto llmRuntime = std::make_unique<mtk::LlmRuntime>();
    auto preloadSw = reinterpret_cast<const mtk::SharedWeightsHandle*>(preloadedSharedWeights);
    const auto status = llmRuntime->initialize(modelOptions, runtimeOptions, preloadSw);
    *runtime = llmRuntime.release();
    return status;
}

void LLM_API mtk_llm_swap_model(void* runtime, const size_t tokenSize, const size_t cacheSize) {
    auto llmRuntime = reinterpret_cast<mtk::LlmRuntime*>(runtime);
    llmRuntime->swapModel(tokenSize, cacheSize);
    return;
}

size_t LLM_API mtk_llm_advance_cache_size(void* runtime) {
    auto llmRuntime = reinterpret_cast<mtk::LlmRuntime*>(runtime);
    return llmRuntime->advanceCacheSize();
}

void LLM_API mtk_llm_release(void* runtime) {
    auto llmRuntime = reinterpret_cast<mtk::LlmRuntime*>(runtime);
    llmRuntime->release();
    delete llmRuntime;
}

void LLM_API mtk_llm_set_medusa_tree_attn(void* runtime, const std::vector<std::vector<int>>& mask,
                                          const std::vector<size_t>& positions) {
    auto llmRuntime = reinterpret_cast<mtk::LlmRuntime*>(runtime);
    llmRuntime->setMedusaTreeAttn(mask, positions);
}

void LLM_API mtk_llm_set_tree_attn(void* runtime, const std::vector<std::vector<int>>& mask,
                                   const std::vector<size_t>& positions,
                                   const size_t modelTokenSize, const size_t tokenOffset) {
    auto llmRuntime = reinterpret_cast<mtk::LlmRuntime*>(runtime);
    llmRuntime->setLlmTreeAttn(mask, positions, modelTokenSize, tokenOffset);
}

void LLM_API mtk_llm_reset_tree_attn(void* runtime) {
    auto llmRuntime = reinterpret_cast<mtk::LlmRuntime*>(runtime);
    llmRuntime->resetLlmTreeAttn();
}

void LLM_API mtk_llm_use_prompt_as_batch_gen(void* runtime) {
    auto llmRuntime = reinterpret_cast<mtk::LlmRuntime*>(runtime);
    llmRuntime->enterFoldedGenBatchMode();
}

void* LLM_API mtk_llm_consume_emb(void* runtime, const char* embBuffer, const size_t embSize,
                                  const LogitsKind outputKind) {
    auto llmRuntime = reinterpret_cast<mtk::LlmRuntime*>(runtime);
    return llmRuntime->consumeEmbeddings(embBuffer, embSize, outputKind);
}

void* LLM_API mtk_llm_inference_once(void* runtime, const std::vector<TokenType>& inputTokens,
                                     const LogitsKind outputKind) {
    auto llmRuntime = reinterpret_cast<mtk::LlmRuntime*>(runtime);
    return llmRuntime->run(inputTokens, outputKind);
}

Batched<void*>
    LLM_API mtk_llm_inference_batch(void* runtime,
                                    const Batched<std::vector<TokenType>>& batchInputTokens,
                                    const LogitsKind outputKind) {
    auto llmRuntime = reinterpret_cast<mtk::LlmRuntime*>(runtime);
    return llmRuntime->runBatched(batchInputTokens, outputKind);
}

std::tuple<void*, void*> // logits, last_hidden_states
    LLM_API mtk_llm_inference_once_return_hidden(void* runtime,
                                                 const std::vector<TokenType>& inputTokens,
                                                 const LogitsKind outputKind) {
    auto llmRuntime = reinterpret_cast<mtk::LlmRuntime*>(runtime);
    auto logits = llmRuntime->run(inputTokens, outputKind);
    auto finalHiddenStates = llmRuntime->getLastHiddenStates();
    if (!llmRuntime->isSeparateLmHead()) {
        LOG(WARN) << "Separated LM Head is not used, so the last hidden states is equivalent to the"
                     " full logits.";
    }
    return {logits, finalHiddenStates};
}

// Return medusa logits
void* LLM_API neuron_medusa_heads_inference_once(void* runtime, void* hiddenState) {
    auto llmRuntime = reinterpret_cast<mtk::LlmRuntime*>(runtime);
    return llmRuntime->runMedusaHeads(hiddenState);
}

void LLM_API mtk_llm_apply_lora(void* runtime, const LoraKey& loraKey) {
    auto llmRuntime = reinterpret_cast<mtk::LlmRuntime*>(runtime);
    llmRuntime->applyLora(loraKey);
}

void LLM_API mtk_llm_apply_lora_from_buffer(void* runtime,
                                            const std::vector<const char*>& loraWeightBuffers,
                                            const std::vector<size_t>& sizes) {
    auto llmRuntime = reinterpret_cast<mtk::LlmRuntime*>(runtime);
    llmRuntime->applyLoraFromBuffer(loraWeightBuffers, sizes);
}

void LLM_API mtk_llm_remove_lora(void* runtime) {
    auto llmRuntime = reinterpret_cast<mtk::LlmRuntime*>(runtime);
    llmRuntime->removeLora();
}

void LLM_API mtk_llm_get_caches(void* runtime, std::vector<std::vector<char*>>& caches,
                                size_t& byteSizePerCache) {
    auto llmRuntime = reinterpret_cast<mtk::LlmRuntime*>(runtime);
    std::vector<std::vector<size_t>> sizes;
    llmRuntime->getKVCaches(&caches, &sizes);
    DCHECK(!sizes.empty() && !sizes.front().empty());
    if (!mtk::llm_helper::allSame(sizes)) {
        LOG(WARN) << "The KV caches shapes/sizes are not all the same.";
    }
    byteSizePerCache = sizes.front().front();
}

std::vector<size_t> LLM_API mtk_llm_get_nopad_caches_nbytes(void* runtime, const size_t numTokens) {
    auto llmRuntime = reinterpret_cast<mtk::LlmRuntime*>(runtime);
    return llmRuntime->getNonPaddedCachesNumBytes(numTokens);
}

size_t LLM_API mtk_llm_save_nopad_caches(void* runtime, const size_t numTokens,
                                         const std::vector<char*>& dstBuffers) {
    auto llmRuntime = reinterpret_cast<mtk::LlmRuntime*>(runtime);
    return llmRuntime->saveNonPaddedCaches(numTokens, dstBuffers);
}

void LLM_API mtk_llm_restore_caches(void* runtime, const size_t numCachedTokens,
                                    const std::vector<const char*>& cacheBuffers,
                                    const std::vector<size_t>& cacheBufferSizes) {
    auto llmRuntime = reinterpret_cast<mtk::LlmRuntime*>(runtime);
    CHECK_EQ(cacheBuffers.size(), cacheBufferSizes.size());

    // Convert to string_view
    std::vector<std::string_view> cacheBufferViews;
    cacheBufferViews.reserve(cacheBuffers.size());
    for (size_t i = 0; i < cacheBuffers.size(); i++) {
        cacheBufferViews.emplace_back(cacheBuffers[i], cacheBufferSizes[i]);
    }
    llmRuntime->restoreKVCaches(numCachedTokens, cacheBufferViews);
}

void LLM_API mtk_llm_reset(void* runtime, const bool resetCache) {
    auto llmRuntime = reinterpret_cast<mtk::LlmRuntime*>(runtime);
    llmRuntime->reset(resetCache);
}

size_t LLM_API mtk_llm_get_per_token_logits_size(void* runtime) {
    auto llmRuntime = reinterpret_cast<mtk::LlmRuntime*>(runtime);
    return llmRuntime->getPerTokenLogitsSize();
}

size_t LLM_API mtk_llm_get_per_token_hidden_states_size(void* runtime) {
    auto llmRuntime = reinterpret_cast<mtk::LlmRuntime*>(runtime);
    return llmRuntime->getPerTokenHiddenStatesSize();
}

size_t LLM_API mtk_llm_get_token_index(void* runtime) {
    auto llmRuntime = reinterpret_cast<mtk::LlmRuntime*>(runtime);
    return llmRuntime->getTokenIndex();
}

void LLM_API mtk_llm_rollback(void* runtime, const size_t rollbackCount) {
    auto llmRuntime = reinterpret_cast<mtk::LlmRuntime*>(runtime);
    llmRuntime->rollback(rollbackCount);
}

void LLM_API mtk_llm_medusa_rollback(void* runtime, const std::vector<size_t>& acceptedIndices) {
    auto llmRuntime = reinterpret_cast<mtk::LlmRuntime*>(runtime);
    llmRuntime->medusaRollback(acceptedIndices);
}

void LLM_API mtk_llm_tree_rollback(void* runtime, const std::vector<size_t>& acceptedIndices,
                                   const size_t relativeBeginOffset) {
    auto llmRuntime = reinterpret_cast<mtk::LlmRuntime*>(runtime);
    llmRuntime->llmTreeRollback(acceptedIndices, relativeBeginOffset);
}

bool LLM_API mtk_llm_is_enabled_cache_eviction(void* runtime) {
    auto llmRuntime = reinterpret_cast<mtk::LlmRuntime*>(runtime);
    return llmRuntime->isEnabledCacheEviction();
}

void LLM_API mtk_llm_hint_num_prompt_tokens(void* runtime, const size_t count) {
    auto llmRuntime = reinterpret_cast<mtk::LlmRuntime*>(runtime);
    llmRuntime->setNumPromptTokensHint(count);
}