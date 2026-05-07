#pragma once

#include "backend/api/neuron/Types.h"
#include "common/file_source.h"
#include "executor/neuron_executor.h"
#include "executor/neuron_usdk_executor.h"
#include "executor/shared_weights.h"
#include "llm_helper/include/cache_eviction.h"
#include "llm_helper/include/lora_weights_loader.h"
#include "llm_helper/include/mask_builder.h"
#include "llm_helper/include/rotary_embedding.h"
#include "mtk_llm.h"
#include "mtk_llm_types.h"

#include <array>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mtk {

using LoraWeightsFileMap = std::unordered_map<LoraKey, FileSource>;

#ifdef USE_USDK_BACKEND
using ExecutorBackend = NeuronUsdkExecutor;
#else
using ExecutorBackend = NeuronExecutor;
#endif

class LlmExecutor : public ExecutorBackend {
public:
    using ShapeValueType = std::remove_extent_t<decltype(RuntimeAPIDimensions::dimensions)>;
    using ShapeType = std::array<ShapeValueType, kDimensionSize>;

    // Dimension where the cache length can be found from the input cache shape
    static constexpr size_t kCacheLengthDim = 2;

    struct RuntimeInfo {
        FileSource modelFile;
        size_t batchSize = 1;
        size_t tokenSize = 1; // E.g. prompt model is 32, whereas generative mode is 1
        size_t cacheSize = 0; // For FMS-based dynamic shape cache
    };

    enum class IOKind {
        Embedding,
        Mask,
        MaskAttn,  // Split Mask: Attention
        MaskCache, // Split Mask: KV Cache
        RotEmb,
        KVCache,
        Lora,
        SharedWeights,
        Logits,

        // Cache Eviction
        Attention,
        SinkRotEmb,

        // Phase 3.4 (deepstack): per-(layer-receiving-injection) ds_padded inputs.
        // Count is set via setDeepstackInputCount() before initialize(); 0 = no deepstack.
        Deepstack,
    };

private:
    static constexpr size_t kUnusedSize = 0;

public:
    // clang-format off
    explicit LlmExecutor(const std::vector<RuntimeInfo>& runtimeInfos,
                         const LlmModelOptions& modelOptions,
                         const LlmRuntimeOptions& runtimeOptions,
                         const SharedWeights& sharedWeights,
                         const size_t numModelChunks,
                         // Rotary Embedding Lut
                         const llm_helper::RotaryEmbeddingMasterLut* rotEmbMasterLut,
                         // Lora
                         const LoraWeightsFileMap& loraWeightsFileMap,
                         // Init cache files
                         const FileSource& initCacheFile,
                         // Inputs
                         const size_t maskInputIndex = 1,
                         const size_t rotEmbInputIndex = 2)
        : ExecutorBackend(getModelFiles(runtimeInfos), sharedWeights),
          kRuntimeInfos(runtimeInfos),
          // Llm specific options
          kMaxTokenLength(modelOptions.maxTokenLength),
          mCacheLength(modelOptions.cacheSize),
          kCacheCount(2 * modelOptions.numLayer / numModelChunks),
          kCacheTypeSize(getLLMTypeSize(modelOptions.cacheType)),
          kMaskType(modelOptions.maskType),
          kMaskTypeSize(getLLMTypeSize(modelOptions.maskType)),
          kUseSplitMask(modelOptions.splitMask),
          kInitTokenIndex(runtimeOptions.startTokenIndex),
          kInitCacheFile(initCacheFile),
          mRotEmbMasterLut(rotEmbMasterLut),
          kRotEmbInputCount(modelOptions.rotEmbNumInputs),
          // Lora Input Weights, infer the number of Lora inputs from bin header if not specified.
          kLoraWeightsFileMap(loraWeightsFileMap),
          kDefaultLoraKey(runtimeOptions.initWithLoraKey),
          kLoraInputCount(getLoraInputCount(runtimeOptions, kLoraWeightsFileMap)),
          kAttnOutputCount(getAttnOutputCount(modelOptions, numModelChunks)),
          kNumHead(modelOptions.numHead),
          kCacheEvictionOptions(runtimeOptions.cacheEvictionOptions) {}

    // clang-format on

    ~LlmExecutor() {}

    // Initialization
    virtual void initialize() override;
    virtual void preInitBufferProcess() override;
    virtual void assignBufferSizesToMax() override;

    virtual void runInferencePrologue() override;

    virtual void runInferenceEpilogue() override;

    // Hot-swap to model with tokenSize and cacheSize if available.
    // Returns true if swap successfully, false if otherwise.
    bool hotSwapModel(const size_t tokenSize, const size_t cacheSize = kUnusedSize);

    // Phase 3.4 (deepstack): set the number of trailing deepstack inputs the model
    // expects (0 = no deepstack). Must be called BEFORE initialize() so that
    // defineIOs() can size the IO table correctly. For Qwen3-VL = 3.
    void setDeepstackInputCount(const size_t n) { mDeepstackInputCount = n; }
    size_t getDeepstackInputCount() const { return mDeepstackInputCount; }
    // Returns the absolute input indices of the 3 ds_padded ports (indices
    // [actualNumInputs - mDeepstackInputCount, actualNumInputs)).
    std::vector<size_t> getDeepstackInputIndexes() const {
        return getInputIndexes(IOKind::Deepstack);
    }

    // Get the next available larger cache size given a token size.
    // Or return the smallest cache size if all cache sizes under given token size are smaller than
    // the current one, assuming the cache will be reset.
    size_t getNextAvailCacheSize(const size_t tokenSize);

    // Get the next available larger cache size, under the same token size.
    size_t getNextAvailCacheSize();

    // Caches
    virtual void initCache();

    size_t getNumCacheBuffers() const;

    // Get cache buffers
    // Some models use different shapes for KV caches in different layers, so in that case
    // `getCacheBuffersWithSizes` needs to be used instead as `getCacheBuffersWithSize` would be
    // inaccurate since it only consider the size/shape of the first cache.
    void getCacheBuffersWithSize(std::vector<char*>& cacheBuffers, size_t& nBytesPerCache);
    void getCacheBuffersWithSizes(std::vector<char*>& cacheBuffers,
                                  std::vector<size_t>& nBytesPerCache);

    std::vector<size_t> getNonPaddedCacheBuffersSizes(const size_t numTokens);

    size_t saveNonPaddedCacheBuffers(const size_t numTokens,
                                     const std::vector<char*>& cacheBuffers);

    // Restore cache from buffers, where the buffers can be either a single concatenated buffer
    // or one buffer per cache input.
    virtual void restoreCaches(const size_t numTokens,
                               const std::vector<std::string_view>& cacheBuffers);

    // Token index
    virtual void resetTokenIndex(const size_t tokenIndex);
    void resetTokenIndex();                 // Reset using kInitTokenIndex
    void setTokenIndex(const size_t index); // NOTE: Need to modify cache if token index was not 0
    void advanceTokenIndex(const int count);
    size_t getTokenIndex() const;
    size_t getEffectiveTokenIndex() const;
    size_t getEffectiveTokenIndex(const size_t tokenIndex) const;
    size_t getNumCachedTokens() const;

    // Align the model state (cache & token index) with the current input. Used for >1t model.
    // Returns the number of tokens being shifted/rolledback
    int alignInputTokens(const size_t numInputToken);
    int alignInputTokens(const size_t numInputToken, const size_t baseModelTokenSize);
    int alignInputTokens(const std::vector<size_t>& acceptedIndices);

    void updatePosEmbAndMask(const size_t numInputToken = 1);

    // Padding
    void setLeftPadding(const size_t leftPadSize);
    void setRightPadding(const size_t rightPadSize);
    void paddingPostprocess(); // General padding postprocessing and will call L/R specific routine

    // Get expected input token count from the model
    size_t getModelTokenSize() const { return mModelTokenSize; }
    // Get expected input token count excluding padded tokens from the model
    size_t getValidModelNumInputToken() const { return mModelTokenSize - getPadSize(); }

    // LoRA-as-inputs
    // Apply Lora based on predefined Lora Key. Empty key will remove Lora and use base weights only
    void applyLoraWeights(const LoraKey& loraKey = "");

    // Apply Lora based on provided Lora weights, will override/bypass any predefined Lora keys.
    void applyLoraWeights(const std::vector<const char*>& loraWeights,
                          const std::vector<size_t>& sizes);

    // Remove Lora and use base weights only
    void removeLoraWeights();

    size_t getCacheLength() const { return mCacheLength; }

    bool isCacheEmpty() const { return getTokenIndex() == 0; }

    // Check if the executor is in folded gen batch mode.
    bool isFoldedGenBatchMode() const;

    // Enter folded gen batch mode by setting folded batch token size
    // The folded gen batch mode can only be exited/disabled by calling `resetTokenIndex()`.
    void enterFoldedGenBatchMode();

    // Rollback tree cache
    void rollbackTreeCache(const std::vector<size_t>& acceptedIndices, size_t baseTokenSize = 0);

    // Set self-defined tree mask
    void setTreeAttn(const std::vector<std::vector<int>>& mask,
                     const std::vector<size_t>& positions, const size_t modelTokenSize,
                     const size_t tokenOffset = 0);
    // Unset self-defined tree mask
    void resetTreeAttn();

    // Wrappers to call IO getters with IOKind and pos instead of raw IO index
#define __DEF__(InOrOut, IOGetterFunc)                                     \
    using Executor::IOGetterFunc;                                          \
    decltype(auto) IOGetterFunc(const IOKind kind, const size_t pos = 0) { \
        return Executor::IOGetterFunc(get##InOrOut##putIndex(kind, pos));  \
    }
    __DEF__(In, getInput)
    __DEF__(In, getInputBuffer)
    __DEF__(In, getInputBufferSizeBytes)
    __DEF__(In, getModelInputSizeBytes)
    __DEF__(Out, getOutput)
    __DEF__(Out, getOutputBuffer)
    __DEF__(Out, getOutputBufferSizeBytes)
    __DEF__(Out, getModelOutputSizeBytes)
#undef __DEF__

protected:
    // IO Setup and Query
    virtual void defineIOs();

    const std::vector<size_t>& getCacheInputIdxs() const {
        return getInputIndexes(IOKind::KVCache);
    }

    const std::vector<size_t>& getCacheOutputIdxs() const {
        return getOutputIndexes(IOKind::KVCache);
    }

    size_t getPadSize() const { return mCurrentPadSize; }
    size_t getLeftPadding() const;
    size_t getRightPadding() const;

    // Cache post-processing specific to left/right padding
    virtual void leftPaddingCachePostprocess();
    virtual void rightPaddingCachePostprocess();

    virtual void linkCacheIOs();

    virtual void rollbackCache(const size_t tokenCount);

    virtual std::vector<char*> getCacheBuffers();

    // Duplicate input caches to all batches during model swap to larger batch size
    virtual void inputCacheDupAllBatches();

    // Override the input batch dim. LLM model input 0 batch dim is 1.
    virtual size_t getInputBatchDim() const override { return 1; }

    // In-place reshape cache inputs from [..., oldCacheLength, ...] to [..., newCacheLength, ...]
    // by padding (upsize) or truncation (downsize).
    virtual void reconstructCaches(const size_t oldCacheLength, const size_t newCacheLength);

    // Helper functions

    // Returns the number of rows per cache input. The batch dimension is included in the calc.
    size_t getCacheNumRows(const size_t index) const;

    // Returns the cache stride size (aka headDim) in bytes, and assumed to be same across caches.
    size_t getCacheStrideSize() const;

public:
    //===---------------===//
    // Cache Eviction APIs
    //===---------------===//

    // Returns a vector of attention weights buffers,
    // or returns an empty vector if there is no attention output defined.
    std::vector<std::string_view> getAttnWeightsBuffers() const;

    // Returns the first attention weights output shape.
    ShapeType getFirstAttnWeightsShape() const;

    void initCacheEviction();

    // Check if cache eviction is being used
    bool isEnabledCacheEviction() const;

    // Check if cache eviction graph is expected
    bool isCacheEvictionGraph() const;

    void setNumPromptTokensHint(const size_t count);

protected:
    // Function to invoke cache eviction.
    // Returns the overall shift in cache due to eviction for right-flush cache format.
    int runCacheEviction(const bool isCacheUpdated);

private:
    void updateCacheMaskBroadcastOption();

    void updateMaskBuffers();

    void initMaskBuilder();

    virtual void setPosEmbed();

    // Build the mapping from model info (token size, cache size) to runtime index
    void buildRuntimeIdxMap();

    // Select the model with largest token size with smallest cache size
    void setDefaultModel();

    static std::vector<size_t> getIndexRange(const size_t startIndex, const size_t count) {
        std::vector<size_t> indexes(count);
        size_t counter = startIndex;
        for (auto& idx : indexes) {
            idx = counter++;
        }
        return indexes;
    }

    static std::vector<FileSource> getModelFiles(const std::vector<RuntimeInfo>& runtimeInfos) {
        std::vector<FileSource> modelFiles;
        for (const auto& runtimeInfo : runtimeInfos) {
            modelFiles.push_back(runtimeInfo.modelFile);
        }
        return modelFiles;
    }

    // Returns `loraInputCount` if defined, otherwise work it out from `loraWeightsFileMap`
    static size_t getLoraInputCount(const LlmRuntimeOptions& runtimeOptions,
                                    const LoraWeightsFileMap& loraWeightsFileMap) {
        if (runtimeOptions.loraInputCount) {
            return runtimeOptions.loraInputCount;
        }
        std::unordered_set<size_t> loraInputsCountSet;
        for (const auto& [loraKey, loraFile] : loraWeightsFileMap) {
            CHECK(loraFile.valid());
            const auto numLoraInputs = llm_helper::LoraWeightsLoader(loraFile).getNumLoraInputs();
            LOG(DEBUG) << " Lora weights '" << loraKey << "' has " << numLoraInputs << " inputs.";
            loraInputsCountSet.insert(numLoraInputs);
        }
        if (loraInputsCountSet.size() > 1) {
            LOG(ERROR) << "Unsupported: Different Lora weight input count found across Lora "
                       << "weights bin files.";
        }
        return loraInputsCountSet.empty() ? 0 : *loraInputsCountSet.cbegin();
    }

    static size_t getAttnOutputCount(const LlmModelOptions& modelOptions, const size_t numChunks) {
        const auto numAttnOutputs = modelOptions.numAttnOutputs;
        const auto numLayer = modelOptions.numLayer;
        if (numAttnOutputs > 1 && numAttnOutputs == numLayer) {
            DCHECK_EQ(numAttnOutputs % numChunks, 0);
            return numAttnOutputs / numChunks;
        }
        return numAttnOutputs;
    }

protected:
    // The number of input tokens the the fixed-shape model takes
    size_t mModelTokenSize = 1;

    const std::vector<RuntimeInfo> kRuntimeInfos;

    // Map [tokenSize][cacheSize] -> runtime index
    std::unordered_map<int, std::unordered_map<int, size_t>> mRuntimeIdxMap;

    // Map tokenSize -> batchSize // NOTE: Assume batch size depends on token size.
    std::unordered_map<size_t, size_t> mBatchSizeMap;

    // Num prompt token recorded for folded gen batch mode
    size_t mGenBatchNumPromptTokens = 0;

    // Cache
    std::vector<ShapeType> mCacheShapes;
    size_t mCacheLength;
    const size_t kMaxTokenLength;
    const size_t kCacheCount;
    const size_t kCacheTypeSize; // bytes

    // Mask
    const LLMType kMaskType;
    const size_t kMaskTypeSize; // bytes
    const bool kUseSplitMask;   // Whether to split mask into cache and attn parts
    bool mCacheMaskUseBroadcast = false;

    enum class PaddingMode {
        LEFT,
        RIGHT
    };

    // Padding
    size_t mCurrentPadSize = 0;
    PaddingMode mPaddingMode = PaddingMode::RIGHT;

    const size_t kInitTokenIndex = 0;

    const FileSource kInitCacheFile;

    // Master lookup table for rotary embedding
    const llm_helper::RotaryEmbeddingMasterLut* mRotEmbMasterLut;
    const size_t kRotEmbInputCount;

    // Mask builder
    std::unique_ptr<llm_helper::MaskBuilder> mMaskBuilder;

    size_t mCurrentTokenIndex = 0; // Default init from 0, also can be numSeenToken

    // The number of tokens that are actually in KV Cache.
    // Identical to `mCurrentTokenIndex` when cache eviction is not used.
    size_t mNumCachedTokens = 0;

    // Will be set to false during init and after model swap
    bool mIsMaskUpdatable = false;

    // LoRA-as-inputs
    const LoraWeightsFileMap kLoraWeightsFileMap;
    const size_t kLoraInputCount = 0;
    const LoraKey kDefaultLoraKey;
    LoraKey mCurrentLoraKey;

    // Cache Eviction
    const LlmRuntimeOptions::CacheEvitionOptions kCacheEvictionOptions;
    const size_t kNumHead;
    const size_t kAttnOutputCount;
    size_t mNumPromptTokensHint = 0;

    std::unique_ptr<llm_helper::CacheEvictionAgent> mCacheEvictionAgent;
    static constexpr size_t kSinkRotEmbNumInputs = 4u;

    // Phase 3.4 (deepstack): trailing deepstack input count (set via setter before init).
    size_t mDeepstackInputCount = 0;

    // Self-defined positions
    std::vector<size_t> mTreePositions;
    size_t mTokenSizeOffset;
};

} // namespace mtk