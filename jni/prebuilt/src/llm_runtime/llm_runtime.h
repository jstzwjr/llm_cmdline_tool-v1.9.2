#pragma once

#include "common/thread_pool.h"
#include "mtk_llm_options.h"
#include "mtk_llm_types.h"
#include "tokenizer/tokenizer.h"

#include <string>
#include <unordered_set>
#include <vector>

#define LLM_API __attribute__((visibility("default")))

namespace mtk {

template <typename T>
using Batched = std::vector<T>;

using Tokens = std::vector<Tokenizer::TokenType>;

class SharedWeightsHandle;

//===---------------===//
// Shared Weights APIs
//===---------------===//

// Smart pointer version of preload shared weights, no manual free needed.
std::unique_ptr<SharedWeightsHandle> preloadSharedWeights(const LlmRuntimeOptions& runtimeOptions);

// Raw pointer version of shared weights preloading with manual free.
void preloadSharedWeights(SharedWeightsHandle** sharedWeightsHandle,
                          const LlmRuntimeOptions& runtimeOptions);
void freePreloadedSharedWeights(SharedWeightsHandle* sharedWeightsHandle);

//===-----------------------------------===//
// Advanced Shared Weights Preloading APIs
//===-----------------------------------===//

// Create an empty shared weights handle without any preloading.
std::unique_ptr<SharedWeightsHandle>
createSharedWeightsHandle(const LlmRuntimeOptions& runtimeOptions);

// Select the subset of shared weights to be preloaded within a handle.
//
// If `repeatAllChunks` is set to true, then the given subset indexes are treated as the shared
// weights indexes for the first model chunk, and will repeat for the rest of the model chunks.
void setSharedWeightsPreloadSubset(SharedWeightsHandle* sharedWeightsHandle,
                                   const std::unordered_set<size_t>& subsetIndexes,
                                   const bool repeatAllChunks = false);

// Preload shared weights while reusing any pre-existing shared weights where possible.
//
// If a shared weights is required by multiple handles, then only one buffer will be
// allocated to hold the shared weights, then shared (reused) across the handles.
//
// If `refHandles` argument is provided, then any shared weights preloaded by these reference
// handles will be reused by `preloadHandles` where possible.
void preloadSharedWeightsUnion(const std::vector<SharedWeightsHandle*>& preloadHandles,
                               const std::vector<SharedWeightsHandle*>& refHandles = {});

// Unload a single shared weights buffer within a handle.
void unloadSharedWeights(SharedWeightsHandle* sharedWeightsHandle, const size_t index);

//===------------===//
// LLM Runtime APIs
//===------------===//

namespace llm_helper {
class RotaryEmbeddingMasterLut;
class TokenEmbeddingLut;
}; // namespace llm_helper

class Executor;
class LlmExecutor;

class LLM_API LlmRuntime {
public:
    LlmRuntime();

    virtual ~LlmRuntime();

    virtual bool initialize(const LlmModelOptions& modelOptions,
                            const LlmRuntimeOptions& runtimeOptions,
                            const SharedWeightsHandle* preloadedSharedWeights = nullptr);

    virtual void release();

    // Get text embedding naively without padding
    virtual void* getTextEmbedding(const Tokens& inputTokens);

    // Get text embedding with batch and padding support
    virtual Batched<void*> getTextEmbeddingBatched(const Batched<Tokens>& inputTokens,
                                                   size_t* leftPadSize = nullptr,
                                                   size_t* rightPadSize = nullptr);

    // Run LLM repeatedly until the entire input embeddings are consumed
    virtual void* consumeEmbeddings(const char* embBuffer, const size_t embSize,
                                    const LogitsKind outputKind = LogitsKind::LAST);

    // Run LLM once with input tokens
    virtual void* run(const Tokens& inputTokens, const LogitsKind outputKind = LogitsKind::LAST);

    // Run LLM once with input embeddings
    virtual void* run(const void* inputEmb, const size_t leftPadSize = 0,
                      const size_t rightPadSize = 0,
                      const LogitsKind outputKind = LogitsKind::LAST);

    // Run LLM once with input tokens
    virtual Batched<void*> runBatched(const Batched<Tokens>& batchInputTokens,
                                      const LogitsKind outputKind = LogitsKind::LAST);

    // Run LLM once with input embeddings
    virtual Batched<void*> runBatched(const void* inputEmb, const size_t leftPadSize = 0,
                                      const size_t rightPadSize = 0,
                                      const LogitsKind outputKind = LogitsKind::LAST);

    void reset(const bool resetCache = true);

    void rollback(const size_t count);

    void swapModel(const size_t tokenSize, const size_t cacheSize = 0);

    size_t advanceCacheSize();

    void enterFoldedGenBatchMode();

    size_t getPerTokenLogitsSize() const;

    size_t getPerTokenHiddenStatesSize() const;

    size_t getTokenIndex() const;

    bool isSeparateLmHead() const; // Check whether separate LmHead is used.

    void* getLastHiddenStates(); // Returns full logits if LmHead is not used.

    void getKVCaches(std::vector<std::vector<char*>>* kvCaches,
                     std::vector<std::vector<size_t>>* kvCacheSizeBytes) const;

    // Get the sizes required to store the non-padded caches to store a specified number of tokens,
    // that is `numTokens` or the total available number of cached tokens, whichever is smaller.
    std::vector<size_t> getNonPaddedCachesNumBytes(const size_t numTokens) const;

    // Save the non-padded caches to the provided buffers.
    // Returns the number of cached tokens stored in the provided buffer,
    // either `numTokens` or the total available number of cached tokens, whichever is smaller.
    size_t saveNonPaddedCaches(const size_t numTokens, const std::vector<char*>& dstBuffers);

    // Restore the LLM state using the given KV cache and its corresponding number of cached tokens.
    // The given KV cache is either one per DLA chunk (concatenated) or one per cache input (split).
    void restoreKVCaches(const size_t numTokens, const std::vector<std::string_view>& kvCaches);

    void setLlmTreeAttn(const std::vector<std::vector<int>>& mask,
                        const std::vector<size_t>& positions, const size_t modelTokenSize,
                        const size_t tokenOffset = 0);

    void resetLlmTreeAttn();

    void llmTreeRollback(const std::vector<size_t>& acceptedIndices,
                         const size_t relativeBeginOffset);

    //===-----===//
    // LoRA APIs
    //===-----===//

    void applyLora(const LoraKey& loraKey);

    void applyLoraFromBuffer(const std::vector<const char*>& loraWeightBuffers,
                             const std::vector<size_t>& sizes);

    void removeLora();

    //===-------===//
    // Medusa APIs
    //===-------===//

    void setMedusaTreeAttn(const std::vector<std::vector<int>>& mask,
                           const std::vector<size_t>& positions);

    void* runMedusaHeads(const void* hiddenState);

    void medusaRollback(const std::vector<size_t>& acceptedIndices);

    //===---------------===//
    // Cacge Eviction APIs
    //===---------------===//

    bool isEnabledCacheEviction() const;

    void setNumPromptTokensHint(const size_t count);

protected:
    //===------------===//
    // Helper Functions
    //===------------===//

    static bool loadBackendLibs();

    LlmExecutor* getFirstLlmExecutor();

    LlmExecutor* getFinalLlmExecutor();

    const LlmExecutor* getFirstLlmExecutor() const;

    const LlmExecutor* getFinalLlmExecutor() const;

    size_t getEffectiveBatchSize() const;

    size_t getEffectiveTokenSize() const;

    Batched<void*> getLogitsBuffer(Executor* executor, const size_t execNumTokens,
                                   const size_t rightPadSize, const LogitsKind outputKind) const;

    virtual bool isLeftPadAllowed() const;

protected:
    std::vector<std::unique_ptr<LlmExecutor>> mLlmDlaExecutors;

    std::unique_ptr<Executor> mDlaLmHeadExecutor;

    std::unique_ptr<Executor> mDlaMedusaHeadsExecutor;

    std::unique_ptr<llm_helper::TokenEmbeddingLut> mTokenEmbLut;

    std::unique_ptr<llm_helper::RotaryEmbeddingMasterLut> mRotEmbMasterLut;

    std::unique_ptr<SharedWeightsHandle> kSharedWeightsHandle; // Used if not preloaded

    // Thread pool used during inference to run prologue and epilogue
    ThreadPool mInferenceThreadPool;

    LlmRuntimeOptions mOptions;
};

} // namespace mtk