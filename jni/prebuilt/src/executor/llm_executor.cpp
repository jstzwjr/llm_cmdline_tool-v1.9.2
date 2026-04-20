#include "executor/llm_executor.h"

#include "common/logging.h"
#include "common/scope_profiling.h"
#include "common/thread_pool.h"
#include "executor/io_buffer_inflator.h"
#include "llm_helper/include/cache_eviction.h"
#include "llm_helper/include/rotary_embedding.h"
#include "llm_helper/include/utils.h"

#include <stdint.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <set>
#include <thread>
#include <vector>

namespace mtk {

// clang-format off

void LlmExecutor::initialize() {
    DLOG_FUNC_LATENCY(s);

    buildRuntimeIdxMap();
    setDefaultModel();
    defineIOs();

    if (isSharedWeightsUsed()) {
        const auto numSharedWeights = numSharedWeightsUsed();
        // Load shared weights and model in parallel
        DCHECK(isInputDefined(IOKind::SharedWeights));
        const auto firstSharedWeightsInputIdx = getInputIndex(IOKind::SharedWeights, 0);
        // Reserve to prevent `initBuffer()` from allocating in case it is called first
        for (size_t counter = 0; counter < numSharedWeights; counter++) {
            reserveInputBuffer(firstSharedWeightsInputIdx + counter);
        }
        initAllocator(); // Allocator must be ready before loadSharedWeights is invoked
        std::thread loadSharedWeightsThread([&] {
            ExecutorBackend::loadSharedWeights(firstSharedWeightsInputIdx);
        });
        std::thread initializeThread([&] {
            ExecutorBackend::initialize();
        });
        loadSharedWeightsThread.join();
        initializeThread.join();
    } else {
        ExecutorBackend::initialize(); // Not using shared weights
    }

    initMaskBuilder();
    initCache();
    applyLoraWeights(kDefaultLoraKey);

    initCacheEviction();
}

// clang-format on

void LlmExecutor::defineIOs() {
    LOG(DEBUG) << "Expects a cache eviction graph: " << isCacheEvictionGraph();

    auto defineInputForMask = [&] {
        if (kUseSplitMask) {
            defineInput(IOKind::MaskAttn);
            defineInput(IOKind::MaskCache);
        } else {
            defineInput(IOKind::Mask);
        }
    };

    auto defineInputForRotEmb = [&] {
        if (isCacheEvictionGraph() && kCacheEvictionOptions.useSinkRotEmb) {
            defineInput(IOKind::SinkRotEmb, kSinkRotEmbNumInputs);
        } else {
            defineInput(IOKind::RotEmb, kRotEmbInputCount);
            DCHECK_NE(kRotEmbInputCount, kSinkRotEmbNumInputs);
        }
    };

    // Reset ordering
    resetInputOrder();
    resetOutputOrder();

    // Define input order
    defineInput(IOKind::Embedding);
    defineInputForMask();
    defineInputForRotEmb();
    defineInput(IOKind::KVCache, kCacheCount);
    defineInput(IOKind::Lora, kLoraInputCount);
    defineInput(IOKind::SharedWeights, numSharedWeightsUsed());

    // Define output order
    defineOutput(IOKind::Logits);
    defineOutput(IOKind::KVCache, kCacheCount);
    defineOutput(IOKind::Attention, isCacheEvictionGraph() ? kAttnOutputCount : 0);

    LOG(DEBUG) << "Input ordering: " << mInputs.getKindOrdering();
    LOG(DEBUG) << "Output ordering: " << mOutputs.getKindOrdering();
}

void LlmExecutor::buildRuntimeIdxMap() {
    size_t runtimeIdx = 0;
    for (const auto& runtimeInfo : kRuntimeInfos) {
        const auto& tokenSize = runtimeInfo.tokenSize;
        const auto& cacheSize = runtimeInfo.cacheSize;
        mRuntimeIdxMap[tokenSize][cacheSize] = runtimeIdx++;

        // Build batch size map
        const auto& batchSize = runtimeInfo.batchSize;
        mBatchSizeMap[tokenSize] = batchSize;
    }
}

void LlmExecutor::setDefaultModel() {
    // Select the model with largest token size with smallest cache size

    auto keyLessThan = [](const auto& pairA, const auto& pairB) {
        return pairA.first < pairB.first;
    };
    auto getMaxKey = [&](const auto& map) {
        return std::max_element(map.begin(), map.end(), keyLessThan)->first;
    };
    auto getMinKey = [&](const auto& map) {
        return std::min_element(map.begin(), map.end(), keyLessThan)->first;
    };

    const auto maxTokenSize = getMaxKey(mRuntimeIdxMap);
    const auto minCacheSize = getMinKey(mRuntimeIdxMap[maxTokenSize]);
    const auto defaultRuntimeIndex = mRuntimeIdxMap[maxTokenSize][minCacheSize];
    this->setDefaultRuntimeIndex(defaultRuntimeIndex);

    mModelTokenSize = maxTokenSize;
    mCacheLength = minCacheSize;

    this->mModelBatchSize = mBatchSizeMap[maxTokenSize];
    LOG(DEBUG) << "Default model batch size = " << this->mModelBatchSize;
}

void LlmExecutor::applyLoraWeights(const LoraKey& loraKey) {
    if (mCurrentLoraKey == loraKey) {
        return; // Already applied
    } else if (loraKey.empty()) {
        removeLoraWeights(); // Empty key, so clear Lora weights to zeros to use base weights
        return;
    } else if (kLoraWeightsFileMap.find(loraKey) == kLoraWeightsFileMap.end()) {
        LOG(ERROR) << "Invalid LoraKey: " << loraKey;
        return;
    }
    std::vector<void*> loraInputBuffers;
    std::vector<size_t> loraInputBufferSizes;
    for (const auto loraInputIdx : getInputIndexes(IOKind::Lora)) {
        const auto& input = this->getInput(loraInputIdx);
        loraInputBuffers.push_back(input.buffer);
        loraInputBufferSizes.push_back(input.usedSizeBytes);
    }
    CHECK_EQ(kLoraInputCount, loraInputBuffers.size());
    llm_helper::LoraWeightsLoader loader(kLoraWeightsFileMap.at(loraKey));
    loader.loadLoraWeights(loraInputBuffers, loraInputBufferSizes);

    mCurrentLoraKey = loraKey;
    LOG(DEBUG) << "Successfully applied Lora weights with key: " << loraKey;
}

void LlmExecutor::applyLoraWeights(const std::vector<const char*>& loraWeights,
                                   const std::vector<size_t>& sizes) {
    CHECK_EQ(kLoraInputCount, loraWeights.size());
    CHECK_EQ(sizes.size(), loraWeights.size());
    DCHECK_EQ(kLoraInputCount, getNumInputsFor(IOKind::Lora));
    for (size_t i = 0; i < kLoraInputCount; i++) {
        auto& input = getInputFor(IOKind::Lora, i);
        const auto loraWeight = loraWeights[i];
        const auto loraWeightSize = sizes[i];
        CHECK_LE(loraWeightSize, input.sizeBytes)
            << "Insufficient buffer allocation (size=" << input.sizeBytes << ") to load Lora input "
            << i << " weights (size=" << loraWeightSize << ")";
        if (loraWeightSize != input.usedSizeBytes) {
            LOG(WARN) << "Expected Lora input " << i << " size by model (" << input.usedSizeBytes
                      << ") != " << "provided Lora weights size (" << loraWeightSize << ")";
        }
        std::memcpy(input.buffer, loraWeight, loraWeightSize);
    }
    mCurrentLoraKey = ""; // Not using any predefined Lora keys
    LOG(DEBUG) << "Successfully applied Lora weights from user provided buffers";
}

void LlmExecutor::removeLoraWeights() {
    // Memset Lora input buffers to zeros
    for (const auto idx : getInputIndexes(IOKind::Lora)) {
        auto& input = this->getInput(idx);
        std::memset(input.buffer, 0, input.usedSizeBytes);
    }
    mCurrentLoraKey = "";
    LOG(DEBUG) << "Removed Lora weights";
}

void LlmExecutor::preInitBufferProcess() {
    ExecutorBackend::preInitBufferProcess();

    // Get input cache shape and ensure cache size is correct
    const auto& cacheInputIdxs = this->getCacheInputIdxs();
    const auto numInputCaches = cacheInputIdxs.size();
    DCHECK_GT(numInputCaches, 0);
    DCHECK_EQ(numInputCaches, kCacheCount);
    mCacheShapes.resize(numInputCaches);
    for (size_t i = 0; i < numInputCaches; i++) {
        auto& cacheShape = mCacheShapes[i];
        this->getRuntimeInputShape(cacheInputIdxs[i], cacheShape.data());
        CHECK_EQ(cacheShape[kCacheLengthDim], getCacheLength())
            << "Please ensure the cache size option is set correctly.";
    }

    // Ensure all stride sizes are the same across cache inputs
    auto getStrideSize = [this](const auto& cacheShape) {
        return llm_helper::reduce_prod(
            cacheShape.begin() + kCacheLengthDim + 1, cacheShape.end(), kCacheTypeSize);
    };
    const auto firstStrideSize = getStrideSize(mCacheShapes[0]);
    for (const auto& cacheShape : mCacheShapes) {
        CHECK_EQ(firstStrideSize, getStrideSize(cacheShape))
            << "Different stride size across caches are not supported.";
    }

    // Verify cache type size using the first cache
    const auto inputCacheSizeBytes = this->getModelInputSizeBytes(IOKind::KVCache, 0);
    const auto inputCacheSize = llm_helper::reduce_prod(mCacheShapes[0]);
    const auto modelCacheTypeSize = inputCacheSizeBytes / inputCacheSize;
    CHECK_EQ(kCacheTypeSize, modelCacheTypeSize)
        << "Mismatch between user provided cache type size (" << kCacheTypeSize << ") "
        << "and actual model cache type size (" << modelCacheTypeSize << ")";

    // Check number of IOs
    CHECK_EQ(getNumInputs(), this->getRuntimeNumInputs())
        << "Number of inputs does not match, please ensure the model is correct.";
    CHECK_EQ(getNumOutputs(), this->getRuntimeNumOutputs())
        << "Number of outputs does not match, please ensure the model is correct.";

    // Link cache IOs
    linkCacheIOs();
}

void LlmExecutor::updateCacheMaskBroadcastOption() {
    if (!kUseSplitMask) {
        mCacheMaskUseBroadcast = false;
        return;
    }
    // NOTE: Can only get the mask broadcast option in non-1t model.
    uint32_t cacheMaskShape[kDimensionSize];
    getRuntimeInputShape(getInputIndex(IOKind::MaskCache), cacheMaskShape);
    mCacheMaskUseBroadcast = (cacheMaskShape[2] == 1 && getModelTokenSize() > 1);
}

void LlmExecutor::updateMaskBuffers() {
    DCHECK(mMaskBuilder);
    if (kUseSplitMask) {
        DCHECK(hasInput(IOKind::MaskAttn));
        DCHECK(hasInput(IOKind::MaskCache));
        const auto& attnMaskInput = getInput(IOKind::MaskAttn);
        const auto& cacheMaskInput = getInput(IOKind::MaskCache);
        mMaskBuilder->setAttnMaskBuffer(attnMaskInput.buffer, attnMaskInput.usedSizeBytes);
        mMaskBuilder->setCacheMaskBuffer(cacheMaskInput.buffer, cacheMaskInput.usedSizeBytes);
        mMaskBuilder->setCacheMaskBroadcastOption(mCacheMaskUseBroadcast);
    } else {
        DCHECK(hasInput(IOKind::Mask));
        const auto& mergedMaskInput = getInput(IOKind::Mask);
        mMaskBuilder->setMaskBuffer(mergedMaskInput.buffer, mergedMaskInput.usedSizeBytes);
    }
}

void LlmExecutor::initMaskBuilder() {
    mMaskBuilder = std::make_unique<llm_helper::MaskBuilder>(kMaskType, getCacheLength());
    updateMaskBuffers();
    mMaskBuilder->buildMask(mModelTokenSize, mCurrentTokenIndex);
    // Duplicate masks for all batches
    if (kUseSplitMask) {
        this->inputDupAllBatches(getInputIndex(IOKind::MaskCache));
        this->inputDupAllBatches(getInputIndex(IOKind::MaskAttn));
    } else {
        this->inputDupAllBatches(getInputIndex(IOKind::Mask));
    }
}

void LlmExecutor::assignBufferSizesToMax() {
    // =============================================================
    //         Input/            |        Size Dependencies
    //         Output            | BatchSize | TokenSize | CacheSize
    // ==========================+===========+===========+===========
    //  Embedding input          |     Y     |     Y     |
    //  Mask input               | (Non-standard size calculation)
    //  Rotary Embdding inputs   |     Y     |     Y     |
    //  Cache inputs             |     Y     |           |     Y
    //  Embedding/Logits output  |     Y     |     Y     |
    //  Cache outputs            |     Y     |           |     Y
    //  Cache outputs (w/ RB)    |     Y     |     Y     |
    // =============================================================

    const auto curBatchSize = this->getBatchSize();
    const auto curTokenSize = this->getModelTokenSize();
    const auto curCacheSize = this->getCacheLength();

    IoBufferInflator ioBufferInflator(
        kRuntimeInfos, RuntimeInfo{{}, curBatchSize, curTokenSize, curCacheSize});

    auto getTokenSize = [](const auto& runtimeInfo) { return runtimeInfo.tokenSize; };
    auto getCacheSize = [](const auto& runtimeInfo) { return runtimeInfo.cacheSize; };
    auto getTokenPlusCacheSize = [](const auto& runtimeInfo) {
        return runtimeInfo.tokenSize + runtimeInfo.cacheSize;
    };

    // Treat the value returned by sizeGetter as size at dim C, then pad it accordingly.
    auto wrapDimCPad = [](const auto& sizeGetter, const size_t typeSize) {
        return [=](const auto& runtimeInfo) -> size_t {
            return llm_helper::alignHWDimCUpperBound(sizeGetter(runtimeInfo), typeSize);
        };
    };

    auto isCurrentModel = [=](const auto& runtimeInfo) -> bool {
        return runtimeInfo.batchSize == curBatchSize && runtimeInfo.tokenSize == curTokenSize
               && runtimeInfo.cacheSize == curCacheSize;
    };

    // Embedding input: BatchSize & TokenSize
    LOG(DEBUG) << "Finding max buffer size for Embedding input";
    ioBufferInflator.useBatchSize().useTokenSize();
    ioBufferInflator.findMaxSizeScenario();
    ioBufferInflator.inflate(this->getInput(0));
    ioBufferInflator.resetUses();

    // Mask input: Special Handling
    updateCacheMaskBroadcastOption();
    if (isInputDefined(IOKind::Mask)) {
        LOG(DEBUG) << "Finding max buffer size for Mask input";
        ioBufferInflator.useBatchSize().useTokenSize();
        ioBufferInflator.useCustomSize(wrapDimCPad(getTokenPlusCacheSize, kMaskTypeSize));
        ioBufferInflator.findMaxSizeScenario();
        ioBufferInflator.inflate(getInputFor(IOKind::Mask));
        ioBufferInflator.resetUses();
    }
    if (isInputDefined(IOKind::MaskCache)) {
        LOG(DEBUG) << "Finding max buffer size for Cache Mask input";
        auto getCacheMaskTokenSize = [&](const auto& runtimeInfo) -> size_t {
            // Only has information of the currently loaded model,
            // so assume all other models don't use broadcast cache mask to calculate upperbound.
            if (isCurrentModel(runtimeInfo) && mCacheMaskUseBroadcast)
                return 1;
            return runtimeInfo.tokenSize;
        };
        ioBufferInflator.useBatchSize();
        if (!mCacheMaskUseBroadcast) {
            ioBufferInflator.useTokenSize();
        }
        ioBufferInflator.useCustomSize(wrapDimCPad(getCacheSize, kMaskTypeSize));
        ioBufferInflator.useCustomSize(getCacheMaskTokenSize);
        ioBufferInflator.findMaxSizeScenario();
        ioBufferInflator.inflate(getInputFor(IOKind::MaskCache));
        ioBufferInflator.resetUses();
    }
    if (isInputDefined(IOKind::MaskAttn)) {
        LOG(DEBUG) << "Finding max buffer size for Attention Mask input";
        ioBufferInflator.useBatchSize().useTokenSize();
        ioBufferInflator.useCustomSize(wrapDimCPad(getTokenSize, kMaskTypeSize));
        ioBufferInflator.findMaxSizeScenario();
        ioBufferInflator.inflate(getInputFor(IOKind::MaskAttn));
        ioBufferInflator.resetUses();
    }

    // Rotary Embdding inputs: BatchSize & TokenSize
    LOG(DEBUG) << "Finding max buffer size for Rotary Embedding input";
    const auto numRotEmbInputs = getNumDefinedInputsFor(IOKind::RotEmb);
    ioBufferInflator.useBatchSize().useTokenSize();
    ioBufferInflator.findMaxSizeScenario();
    for (size_t pos = 0; pos < numRotEmbInputs; pos++) {
        ioBufferInflator.inflate(getInputFor(IOKind::RotEmb, pos));
    }
    ioBufferInflator.resetUses();

    // Cache Eviction Rotary Embedding inputs
    if (isInputDefined(IOKind::SinkRotEmb)) {
        DCHECK_EQ(getNumDefinedInputsFor(IOKind::SinkRotEmb), 4);
        // Q trivially depends on BatchSize & TokenSize
        ioBufferInflator.useBatchSize().useTokenSize();
        ioBufferInflator.findMaxSizeScenario();
        ioBufferInflator.inflate(getInputFor(IOKind::SinkRotEmb, 0)); // Q Cos
        ioBufferInflator.inflate(getInputFor(IOKind::SinkRotEmb, 1)); // Q Sin
        ioBufferInflator.resetUses();
        // K depends on BatchSize and (CacheSize + TokenSize)
        ioBufferInflator.useBatchSize().useCustomSize(getTokenPlusCacheSize);
        ioBufferInflator.findMaxSizeScenario();
        ioBufferInflator.inflate(getInputFor(IOKind::SinkRotEmb, 2)); // K Cos
        ioBufferInflator.inflate(getInputFor(IOKind::SinkRotEmb, 3)); // K Sin
        ioBufferInflator.resetUses();
    }

    // Cache inputs: BatchSize & CacheSize
    LOG(DEBUG) << "Finding max buffer size for Cache input";
    ioBufferInflator.useBatchSize().useCacheSize();
    ioBufferInflator.findMaxSizeScenario();
    for (const auto cacheInputIdx : getCacheInputIdxs()) {
        ioBufferInflator.inflate(this->getInput(cacheInputIdx));
    }
    ioBufferInflator.resetUses();

    // Embedding/Logits output: BatchSize & TokenSize
    LOG(DEBUG) << "Finding max buffer size for Embedding/Logits output";
    ioBufferInflator.useBatchSize().useTokenSize();
    ioBufferInflator.findMaxSizeScenario();
    ioBufferInflator.inflate(this->getOutput(0));
    ioBufferInflator.resetUses();

    // Cache outputs: BatchSize & CacheSize
    LOG(DEBUG) << "Finding max buffer size for Cache output";
    ioBufferInflator.useBatchSize().useCacheSize();
    ioBufferInflator.findMaxSizeScenario();
    for (const auto cacheOutputIdx : getCacheOutputIdxs()) {
        ioBufferInflator.inflate(this->getOutput(cacheOutputIdx));
    }
    ioBufferInflator.resetUses();

    // Attn outputs: Special Handling
    LOG(DEBUG) << "Finding max buffer size for Attn weight";
    const auto numAttnOutputs = getNumOutputsFor(IOKind::Attention);
    if (numAttnOutputs > 0) {
        using AttentionType = int16_t;
        // Reduced attn dim C is 1, whereas normal attn dim C is token_size + cache_size.
        if (!kCacheEvictionOptions.useIngraphAttnReduce) {
            ioBufferInflator.useCustomSize(
                wrapDimCPad(getTokenPlusCacheSize, sizeof(AttentionType)));
        }
        if (kCacheEvictionOptions.mode == CacheEvictionMode::LocalSnapKV) {
            ioBufferInflator.useTokenSize();
        }
        ioBufferInflator.findMaxSizeScenario();
        for (size_t i = 0; i < numAttnOutputs; i++) {
            ioBufferInflator.inflate(getOutputFor(IOKind::Attention, i));
        }
        ioBufferInflator.resetUses();
    }
}

bool LlmExecutor::hotSwapModel(const size_t tokenSize, const size_t cacheSize) {
    DLOG_FUNC_LATENCY(ms)

    // Save old values
    const auto oldRuntimeIdx = this->getRuntimeIndex();
    const size_t oldNumInputToken = mModelTokenSize;

    auto mapHasKey = [](const auto& map, const auto& key) { return map.find(key) != map.end(); };

    if (!mapHasKey(mRuntimeIdxMap, tokenSize)) {
        LOG(ERROR) << "Model swap: No model with tokenSize=" << tokenSize << " is available";
        return false;
    }
    // Search for suitable runtime matching the requirements (token size & cache size)
    const auto& cacheSizeRuntimeMap = mRuntimeIdxMap[tokenSize];
    if (cacheSize != kUnusedSize && !mapHasKey(cacheSizeRuntimeMap, cacheSize)) {
        LOG(ERROR) << "Model swap: No model with tokenSize=" << tokenSize
                   << " has cacheSize=" << cacheSize;
        return false;
    }
    // Maintain the current (old) cache size if not specified in the argument
    const auto oldCacheSize = getCacheLength();
    size_t newCacheSize = (cacheSize == kUnusedSize) ? oldCacheSize : cacheSize;
    if (!mapHasKey(cacheSizeRuntimeMap, newCacheSize)) {
        const auto availableCacheSize = getNextAvailCacheSize(tokenSize);
        LOG(DEBUG) << "The cache size " << newCacheSize << " is not available when switching to "
                   << "token size " << tokenSize
                   << ". Selecting the first available cache size: " << availableCacheSize;
        newCacheSize = availableCacheSize;
    }

    const auto runtimeIdx = cacheSizeRuntimeMap.at(newCacheSize);
    if (runtimeIdx == oldRuntimeIdx) {
        LOG(DEBUG) << "Model swapping to itself.";
        return true;
    }

    this->selectRuntime(runtimeIdx);

    const auto newRuntimeIdx = this->getRuntimeIndex();
    if (oldRuntimeIdx == newRuntimeIdx) {
        LOG(WARN) << "Failed to switch to model with tokenSize=" << tokenSize
                  << " and cacheSize=" << cacheSize << ". Model currently remains at "
                  << "(tokenSize=" << oldNumInputToken << ", cacheSize=" << oldCacheSize
                  << "): " << this->getModelName();
        return false;
    }

    // Update model variables
    // Mask length = cache size (length) + num input token
    mModelTokenSize = tokenSize;

    // Update cache size
    if (oldCacheSize != newCacheSize) {
        LOG(DEBUG) << "Updating cache size from " << oldCacheSize << " to " << newCacheSize;
        DCHECK_EQ(oldCacheSize, getCacheLength());
        reconstructCaches(oldCacheSize, newCacheSize);
        mCacheLength = newCacheSize;
        for (auto& cacheShape : mCacheShapes) {
            DCHECK_LT(kCacheLengthDim, cacheShape.size());
            cacheShape[kCacheLengthDim] = newCacheSize;
        }
        mMaskBuilder->updateCacheLength(newCacheSize);
        mNumCachedTokens = std::min(newCacheSize, mNumCachedTokens); // Ensure at most cache size
    }

    // Update batch size. Currently it is tied to the token size (via mBatchSizeMap).
    const auto oldBatchSize = this->mModelBatchSize;
    const auto newBatchSize = mBatchSizeMap[tokenSize];
    if (oldBatchSize != newBatchSize) {
        LOG(DEBUG) << "Updating batch size from " << oldBatchSize << " to " << newBatchSize;
        this->mModelBatchSize = newBatchSize;
        this->verifyBatchSize();
        // Update cacheShape batch dim (0)
        for (auto& cacheShape : mCacheShapes) {
            DCHECK_EQ(cacheShape.size(), 4);
            cacheShape[0] = newBatchSize;
        }
    }

    // Update IOs if there are changes after model swap
    defineIOs();
    updateModelIO();
    updateCacheMaskBroadcastOption();

    this->registerRuntimeIO(); // Attach IO buffers to model runtime

    // If batch size increases, duplicate input caches
    if (newBatchSize > oldBatchSize) {
        inputCacheDupAllBatches();
    }

    // Rebuild mask because different token/cache size values will produce different mask shapes
    mMaskBuilder->markMaskDirty();

    // Update mask size
    updateMaskBuffers();

    return true;
}

size_t LlmExecutor::getNextAvailCacheSize(const size_t tokenSize) {
    CHECK(mRuntimeIdxMap.find(tokenSize) != mRuntimeIdxMap.end())
        << "The provided token size " << tokenSize << " is not valid.";
    const auto curCacheSize = getCacheLength();
    const auto& cacheSizeRuntimeMap = mRuntimeIdxMap[tokenSize];
    DCHECK(!cacheSizeRuntimeMap.empty());

    std::set<size_t> availableCacheSizes; // Ordered set
    for (const auto& [cacheSize, _] : cacheSizeRuntimeMap) {
        availableCacheSizes.insert(cacheSize);
    }
    DCHECK(!availableCacheSizes.empty());

    LOG(DEBUG) << "Available cache sizes for " << getModelTokenSize() << "t model: "
               << std::vector(availableCacheSizes.begin(), availableCacheSizes.end());

    // Return the min available cache size if even the max cache size is smaller than current one.
    // This happens when resetting to prompt mode.
    const size_t minCacheSize = *availableCacheSizes.cbegin();
    const size_t maxCacheSize = *availableCacheSizes.crbegin();
    if (maxCacheSize < curCacheSize) {
        return minCacheSize;
    }

    // Otherwise, return the next larger cache size that is larger than current one and large enough
    // for the next inference step.
    const size_t minRequiredCacheSize = getNumCachedTokens() + tokenSize;
    auto isSufficient = [=](const size_t cacheSize) {
        return isEnabledCacheEviction() || cacheSize >= minRequiredCacheSize;
    };
    size_t nextLargerCacheSize = curCacheSize;
    for (const auto cacheSize : availableCacheSizes) {
        if (cacheSize > curCacheSize && isSufficient(cacheSize)) {
            nextLargerCacheSize = cacheSize;
            break;
        }
    }
    return nextLargerCacheSize;
}

size_t LlmExecutor::getNextAvailCacheSize() {
    return getNextAvailCacheSize(getModelTokenSize());
}

void LlmExecutor::reconstructCaches(const size_t oldCacheLength, const size_t newCacheLength) {
    DLOG_FUNC_LATENCY(ms)

    if (oldCacheLength == newCacheLength) {
        return; // Do nothing
    }

    const auto& cacheInputIndexes = getCacheInputIdxs();
    const size_t numCacheInputs = cacheInputIndexes.size();
    DCHECK_EQ(mCacheShapes.size(), numCacheInputs);

    const auto strideSize = getCacheStrideSize();
    const auto numCachedTokens = getNumCachedTokens();
    const size_t numCopyTokens =
        std::min(numCachedTokens, std::min(oldCacheLength, newCacheLength));
    const size_t copySize = numCopyTokens * strideSize;

    const size_t oldRowSize = oldCacheLength * strideSize;
    const size_t newRowSize = newCacheLength * strideSize;
    const size_t oldOffset = oldRowSize - copySize;
    const size_t newOffset = newRowSize - copySize;

    if (numCopyTokens < numCachedTokens) {
        LOG(WARN) << "reconstructCaches (" << oldCacheLength << "c -> " << newCacheLength
                  << "c) will truncate " << (numCachedTokens - numCopyTokens)
                  << " tokens from the cache of " << numCachedTokens << " tokens.";
    }

    // Cache size reduction
    auto downsizeOneCache = [&](const size_t index) {
        DCHECK_LT(index, getNumInputsFor(IOKind::KVCache));
        auto& cacheInput = getInputFor(IOKind::KVCache, index);
        const auto numRows = getCacheNumRows(index);
        DCHECK_GE(numRows, 1);

        const auto cacheBuffer = reinterpret_cast<char*>(cacheInput.buffer);

        // Move in order of rows to prevent overwriting data that has yet to move
        // because the destination address is always less than source address.
        for (size_t rowIdx = 0; rowIdx < numRows; rowIdx++) {
            const auto oldCacheBufferRow = cacheBuffer + rowIdx * oldRowSize;
            auto newCacheBufferRow = cacheBuffer + rowIdx * newRowSize;

            // Move cache data to new location
            const auto src = oldCacheBufferRow + oldOffset;
            auto dst = newCacheBufferRow + newOffset;
            std::memmove(dst, src, copySize);

            // Clear off data from original location
            std::memset(oldCacheBufferRow, 0, newOffset);
        }
    };

    // Cache size increase
    auto upsizeOneCache = [&](const size_t index) {
        DCHECK_LT(index, getNumInputsFor(IOKind::KVCache));
        auto& cacheInput = getInputFor(IOKind::KVCache, index);
        const auto numRows = getCacheNumRows(index);
        DCHECK_GE(numRows, 1);

        // Check size availability
        const size_t sizeRequired = numRows * newCacheLength * strideSize;
        const size_t sizeAllocated = cacheInput.sizeBytes;
        if (sizeAllocated < sizeRequired) {
            LOG(ERROR) << "New cache length of " << newCacheLength << " requires buffer size of "
                       << sizeRequired << " but only " << sizeAllocated << " is allocated.";
        }

        const auto cacheBuffer = reinterpret_cast<char*>(cacheInput.buffer);

        // Move in reverse order of rows to prevent overwriting data that has yet to move
        // because the destination address is always greter than source address.
        for (int64_t rowIdx = numRows - 1; rowIdx >= 0; rowIdx--) {
            const auto oldCacheBufferRow = cacheBuffer + rowIdx * oldRowSize;
            auto newCacheBufferRow = cacheBuffer + rowIdx * newRowSize;

            // Move cache data to new location
            const auto src = oldCacheBufferRow + oldOffset;
            auto dst = newCacheBufferRow + newOffset;
            std::memmove(dst, src, copySize);

            // Clear off data from original location
            auto clearStart = src;
            const auto clearEnd = std::min(src + copySize, dst);
            std::memset(clearStart, 0, clearEnd - clearStart);

            // Optional full zeroize approach, but will memset on more values
            // std::memset(newCacheBufferRow, 0, newOffset);
        }
    };

    auto resizeOneCache = [&](const size_t index) -> void {
        return (oldCacheLength > newCacheLength) ? downsizeOneCache(index) : upsizeOneCache(index);
    };

    ThreadPool threadPool(ThreadSpawn::Exp);
    for (size_t i = 0; i < numCacheInputs; i++)
        threadPool.push(resizeOneCache, i);
    threadPool.joinAll();
}

void LlmExecutor::inputCacheDupAllBatches() {
    for (const auto cacheInputIdx : getCacheInputIdxs()) {
        this->inputDupAllBatches(cacheInputIdx);
    }
}

void LlmExecutor::setNumPromptTokensHint(const size_t count) {
    mNumPromptTokensHint = count;
}

bool LlmExecutor::isFoldedGenBatchMode() const {
    return mGenBatchNumPromptTokens != 0;
}

void LlmExecutor::enterFoldedGenBatchMode() {
    if (mModelTokenSize == 1) {
        LOG(DEBUG) << "Ignore setting folded gen batch mode on 1t model.";
        return;
    }
    mGenBatchNumPromptTokens = getTokenIndex();
    mMaskBuilder->enterFoldedGenBatchMode(mGenBatchNumPromptTokens);
}

void LlmExecutor::linkCacheIOs() {
    const auto& cacheInputIndexes = getCacheInputIdxs();
    const auto& cacheOutputIndexes = getCacheOutputIdxs();
    const size_t numCaches = cacheInputIndexes.size();
    for (size_t i = 0; i < numCaches; i++) {
        this->linkModelIO(cacheInputIndexes[i], cacheOutputIndexes[i]);
    }
}

void LlmExecutor::resetTokenIndex(const size_t tokenIndex) {
    mMaskBuilder->reset();
    mGenBatchNumPromptTokens = 0;
    mNumPromptTokensHint = 0;
    setTokenIndex(tokenIndex);
    mNumCachedTokens = tokenIndex;
    if (mCacheEvictionAgent) {
        mCacheEvictionAgent->reset();
    }
}

void LlmExecutor::resetTokenIndex() {
    resetTokenIndex(kInitTokenIndex);
}

void LlmExecutor::setTokenIndex(const size_t index) {
    const auto effectiveTokenIndex = getEffectiveTokenIndex(index);
    if (effectiveTokenIndex >= kMaxTokenLength) {
        LOG(FATAL) << "Attempting to set token index (" << effectiveTokenIndex
                   << ") exceeding the supported max token length (" << kMaxTokenLength << ")";
        return;
    }
    mCurrentTokenIndex = index;
}

void LlmExecutor::advanceTokenIndex(const int count) {
    setTokenIndex(mCurrentTokenIndex + count);
    mNumCachedTokens = std::min(mNumCachedTokens + count, getCacheLength());
}

size_t LlmExecutor::getTokenIndex() const {
    return mCurrentTokenIndex;
}

size_t LlmExecutor::getEffectiveTokenIndex() const {
    return getEffectiveTokenIndex(mCurrentTokenIndex);
}

size_t LlmExecutor::getEffectiveTokenIndex(const size_t tokenIndex) const {
    if (isFoldedGenBatchMode()) {
        // In folded gen batch mode
        CHECK_GE(tokenIndex, mGenBatchNumPromptTokens);
        const auto numGenTokens = tokenIndex - mGenBatchNumPromptTokens;
        CHECK_EQ(numGenTokens % mModelTokenSize, 0);
        const size_t decodingStep = numGenTokens / mModelTokenSize;
        return mGenBatchNumPromptTokens + decodingStep;
    }
    return tokenIndex;
}

size_t LlmExecutor::getNumCachedTokens() const {
    DCHECK_LE(mNumCachedTokens, getTokenIndex());
    DCHECK_LE(mNumCachedTokens, getCacheLength());
    return mNumCachedTokens;
}

int LlmExecutor::alignInputTokens(const size_t numInputToken) {
    int rollbackCount = mModelTokenSize - numInputToken;
    if (rollbackCount > 0) {
        CHECK_GE(mCurrentTokenIndex, rollbackCount) << "Total tok count < model input tok count";
        rollbackCache(rollbackCount);
        LOG(DEBUG) << "Tokens/Caches alignment rollback count = " << rollbackCount;

        advanceTokenIndex(-rollbackCount);

        // Rebuild mask as updateMask requires mCurrentTokenIndex to be monotonically increasing
        mMaskBuilder->markMaskDirty();
    }
    return rollbackCount;
}

int LlmExecutor::alignInputTokens(const size_t numInputToken, const size_t baseModelTokenSize) {
    int rollbackCount;
    if (baseModelTokenSize == 0) {
        rollbackCount = mModelTokenSize - numInputToken;
    } else {
        // draft model will encounter tree rollback issue in spd codebase.
        // mModelTokenSize becomes 1 as draft model is drafting.
        rollbackCount = baseModelTokenSize - numInputToken;
    }
    if (rollbackCount > 0) {
        CHECK_GE(mCurrentTokenIndex, rollbackCount) << "Total tok count < model input tok count";
        rollbackCache(rollbackCount);
        LOG(DEBUG) << "Tokens/Caches alignment rollback count = " << rollbackCount;

        advanceTokenIndex(-rollbackCount);

        // Rebuild mask as updateMask requires mCurrentTokenIndex to be monotonically increasing
        mMaskBuilder->markMaskDirty();
    }
    return rollbackCount;
}

int LlmExecutor::alignInputTokens(const std::vector<size_t>& acceptedIndices) {
    const size_t numInputToken = acceptedIndices.size();
    int rollbackCount = mModelTokenSize - numInputToken;
    if (rollbackCount > 0) {
        CHECK_GE(mCurrentTokenIndex, rollbackCount) << "Total tok count < model input tok count";
        rollbackTreeCache(acceptedIndices);
        LOG(DEBUG) << "Tokens/Caches alignment rollback count = " << rollbackCount;

        advanceTokenIndex(-rollbackCount);

        // Rebuild mask as updateMask requires mCurrentTokenIndex to be monotonically increasing
        mMaskBuilder->markMaskDirty();
    }
    return rollbackCount;
}

void LlmExecutor::runInferencePrologue() {
    ExecutorBackend::runInferencePrologue();

    // NOTE: Padding need to be set before this is called
    updatePosEmbAndMask(mModelTokenSize);
}

void LlmExecutor::runInferenceEpilogue() {
    // Advance token index by the number of valid (non-padded) tokens seen by the model in this step
    advanceTokenIndex(getValidModelNumInputToken());

    // Perform any necessary adjustments when padding is used
    paddingPostprocess();

    ExecutorBackend::runInferenceEpilogue();
}

// Also updates the token index
void LlmExecutor::updatePosEmbAndMask(const size_t numInputToken) {
    const auto effectiveTokenIndex = getEffectiveTokenIndex(mCurrentTokenIndex + numInputToken);
    if (effectiveTokenIndex > kMaxTokenLength) {
        LOG(FATAL) << "Attempting to generate tokens exceeding the supported max token length ("
                   << kMaxTokenLength << ")";
    }
    if (mCurrentTokenIndex > 0 && getLeftPadding() > 0) {
        LOG(FATAL) << "Left-padding is only allowed in the first prompt pass.";
    }
    const auto tokenIndex = isEnabledCacheEviction() ? getNumCachedTokens() : getTokenIndex();
    mMaskBuilder->updateMask(mModelTokenSize, tokenIndex, numInputToken);

    // Duplicate masks for all batches
    if (kUseSplitMask) {
        this->inputDupAllBatches(getInputIndex(IOKind::MaskCache));
        this->inputDupAllBatches(getInputIndex(IOKind::MaskAttn));
    } else {
        this->inputDupAllBatches(getInputIndex(IOKind::Mask));
    }

    setPosEmbed();
    DLOG_FUNC_EXIT
}

void LlmExecutor::setPosEmbed() {
    // Cut the array from master
    const auto effectiveTokenIndex = getEffectiveTokenIndex();
    if (effectiveTokenIndex >= kMaxTokenLength) {
        LOG(FATAL) << "Attempting to set rotaty embedding using index exceeding the supported "
                   << "max token length (" << kMaxTokenLength << ")";
    }

    DLOG_FUNC_LATENCY(ms)

    auto getRotEmbInfo = [this]() -> std::pair<IOKind, size_t> {
        if (hasInput(IOKind::RotEmb)) {
            DCHECK_EQ(getNumInputsFor(IOKind::RotEmb), kRotEmbInputCount);
            return {IOKind::RotEmb, kRotEmbInputCount};
        } else if (hasInput(IOKind::SinkRotEmb)) {
            return {IOKind::SinkRotEmb, kSinkRotEmbNumInputs};
        }
        LOG(FATAL) << "No RotEmb inputs available.";
        return {};
    };

    auto getRotEmbInputs = [&]() {
        const auto [activeRotEmbKind, numRotEmbInputs] = getRotEmbInfo();
        std::vector<void*> rotEmbInputs(numRotEmbInputs);
        for (size_t i = 0; i < numRotEmbInputs; i++)
            rotEmbInputs[i] = getInputBuffer(activeRotEmbKind, i);
        return rotEmbInputs;
    };

    const bool isTreeAttn = !mTreePositions.empty();

    if (isFoldedGenBatchMode()) {
        // In folded gen batch mode
        DCHECK_EQ(getLeftPadding(), 0);
        DCHECK_EQ(getRightPadding(), 0);

        // Get the actual token index
        const auto& realTokenIndex = effectiveTokenIndex;

        // Positions are all zeros (relative to current token index) because they are of the same
        // token position, just in different batches.
        const std::vector<size_t> positions(mModelTokenSize, 0);

        mRotEmbMasterLut->setEmbed(getRotEmbInputs(), realTokenIndex, positions);
    } else if (isTreeAttn) {
        // Get the actual token index
        const auto& realTokenIndex = effectiveTokenIndex;
        const auto tokenIndex = getTokenIndex();
        CHECK_EQ(realTokenIndex, tokenIndex) << "They should be the same";
        // Tree-attention: set mTreePositions as embeddings
        mRotEmbMasterLut->setEmbed(getRotEmbInputs(), tokenIndex, mTreePositions, mTokenSizeOffset,
                                   getLeftPadding(), getRightPadding());
    } else {
        mRotEmbMasterLut->setEmbed(getRotEmbInputs(), getTokenIndex(), mModelTokenSize,
                                   getLeftPadding(), getRightPadding(), getCacheLength());
    }

    // Duplicate for all batches
    for (const auto inputIdx : getInputIndexes(IOKind::RotEmb)) {
        this->inputDupAllBatches(inputIdx);
    }
    for (const auto inputIdx : getInputIndexes(IOKind::SinkRotEmb)) {
        this->inputDupAllBatches(inputIdx);
    }
}

size_t LlmExecutor::getLeftPadding() const {
    return (mPaddingMode == PaddingMode::LEFT) ? mCurrentPadSize : 0;
}

size_t LlmExecutor::getRightPadding() const {
    return (mPaddingMode == PaddingMode::RIGHT) ? mCurrentPadSize : 0;
}

void LlmExecutor::setLeftPadding(const size_t leftPadSize) {
    mCurrentPadSize = leftPadSize;
    mPaddingMode = PaddingMode::LEFT;

    // Notify mask builder about padding
    mMaskBuilder->notifyLeftPadding(leftPadSize);
}

void LlmExecutor::setRightPadding(const size_t rightPadSize) {
    mCurrentPadSize = rightPadSize;
    mPaddingMode = PaddingMode::RIGHT;

    // Notify mask builder about padding
    mMaskBuilder->notifyRightPadding(rightPadSize);
}

void LlmExecutor::paddingPostprocess() {
    if (mCurrentPadSize == 0) {
        return;
    }

    if (mPaddingMode == PaddingMode::RIGHT) {
        rightPaddingCachePostprocess();
    } else if (mPaddingMode == PaddingMode::LEFT) {
        leftPaddingCachePostprocess();
    }

    // Reset padding size
    mCurrentPadSize = 0;
}

void LlmExecutor::leftPaddingCachePostprocess() {
    // NOTE: This part might not actually be needed

    // Stride size is same across caches
    const size_t strideSizeBytes = getCacheStrideSize();
    const size_t rowSize = getCacheLength() * strideSizeBytes;
    const size_t offset = (getCacheLength() - getModelTokenSize()) * strideSizeBytes;
    const size_t zeroCount = getLeftPadding() * strideSizeBytes;

    // Fill padded sections with zeros
    size_t cacheCounter = 0;
    for (const auto cacheInputIdx : getCacheInputIdxs()) {
        auto cacheBuffer = reinterpret_cast<char*>(this->getInputBuffer(cacheInputIdx));
        const size_t numRows = getCacheNumRows(cacheCounter++);
        for (size_t rowIdx = 0; rowIdx < numRows; rowIdx++) {
            auto cacheBufRow = cacheBuffer + rowIdx * rowSize; // Pointer pointing to start of row
            std::memset(cacheBufRow + offset, 0, zeroCount);
        }
    }
}

void LlmExecutor::rightPaddingCachePostprocess() {
    // advanceTokenIndex() has to be called first before calling rollbackCache()
    rollbackCache(mCurrentPadSize);
}

size_t LlmExecutor::getCacheNumRows(const size_t index) const {
    CHECK_GT(mCacheShapes.size(), 0) << "Cache shapes have not been initialized.";
    CHECK_LT(index, mCacheShapes.size());
    const auto& cacheShape = mCacheShapes[index];
    // NOTE: cacheShape[0] is the batch dim
    return llm_helper::reduce_prod(cacheShape.begin(), cacheShape.begin() + kCacheLengthDim);
}

size_t LlmExecutor::getCacheStrideSize() const {
    CHECK_GT(mCacheShapes.size(), 0) << "Cache shapes have not been initialized.";
    const auto& cacheShape = mCacheShapes[0];
    return llm_helper::reduce_prod(
        cacheShape.begin() + kCacheLengthDim + 1, cacheShape.end(), kCacheTypeSize);
}

void LlmExecutor::initCache() {
    DLOG_FUNC_LATENCY(ms)
    resetTokenIndex();
    if (!kInitCacheFile) {
        // Use default zero initialization if no cache file provided
        for (const auto cacheIdx : getCacheInputIdxs()) {
            auto& inputCache = this->getInput(cacheIdx);
            char* cacheBuffer = reinterpret_cast<char*>(inputCache.buffer);
            const size_t cacheSizeBytes = inputCache.sizeBytes;
            std::memset(cacheBuffer, 0, cacheSizeBytes);
        }
        LOG(DEBUG) << "initCache: zero initialization";
        return;
    }

    LOG(DEBUG) << "initCache: precomputed cache initialization";

    if (!kInitCacheFile.valid()) {
        LOG(FATAL) << "Unable to load init cache file: " << kInitCacheFile;
    }
    const auto [cacheFileData, cacheFileSize] = kInitCacheFile.get();

    const auto& cacheInputIdxs = getCacheInputIdxs();
    DCHECK_EQ(cacheInputIdxs.size(), kCacheCount);

    const bool requireAutoPadding = [&, cacheFileSize = cacheFileSize]() -> bool {
        size_t totalCacheBufferSize = 0;
        for (const auto cacheIdx : cacheInputIdxs) {
            totalCacheBufferSize += getModelInputSizeBytes(cacheIdx);
        }
        DCHECK_LE(cacheFileSize, totalCacheBufferSize);
        return cacheFileSize < totalCacheBufferSize;
    }();

    LOG(DEBUG) << "initCache requires auto padding: " << (requireAutoPadding ? "yes" : "no");

    // Get the size of the non-padded region in each cache
    auto getValidCacheSizeBytes = [this](const size_t cacheIdx) -> size_t {
        // KV Cache shape: (1, num_kv_heads, cache_size, head_dim)
        return getCacheNumRows(cacheIdx) * kInitTokenIndex * getCacheStrideSize();
    };

    if (requireAutoPadding) {
        size_t readOffset = 0;
        for (size_t cacheIdx = 0; cacheIdx < kCacheCount; cacheIdx++) {
            const auto cacheInputIdx = cacheInputIdxs[cacheIdx];
            const auto cacheBufferSizeBytes = getModelInputSizeBytes(cacheInputIdx);
            const auto curReadSize = getValidCacheSizeBytes(cacheIdx);
            DCHECK_LT(curReadSize, cacheBufferSizeBytes);
            auto cacheBuffer = getInputBuffer(cacheInputIdx);
            CHECK_LE(readOffset + curReadSize, cacheFileSize)
                << "Total size for cache inputs is larger than the actual size read from the cache "
                << "file (" << cacheFileSize << ")";
            std::memcpy(cacheBuffer, cacheFileData + readOffset, curReadSize);
            readOffset += curReadSize;
        }
        if (readOffset != cacheFileSize) {
            LOG(ERROR) << "Error reading cache file. Total read size is " << readOffset
                       << " but cache file size is " << cacheFileSize;
        }
        DCHECK_LE(kInitTokenIndex, getCacheLength());
        reconstructCaches(kInitTokenIndex, getCacheLength());
    } else {
        size_t readOffset = 0;
        for (const auto cacheInputIdx : cacheInputIdxs) {
            const auto cacheSizeBytes = getModelInputSizeBytes(cacheInputIdx);
            auto cacheBuffer = getInputBuffer(cacheInputIdx);
            CHECK_LE(readOffset + cacheSizeBytes, cacheFileSize)
                << "Total size for cache inputs is larger than the actual size read from the cache "
                   "file"
                << " (" << cacheFileSize << ")";
            std::memcpy(cacheBuffer, cacheFileData + readOffset, cacheSizeBytes);
            readOffset += cacheSizeBytes;
        }
    }
}

size_t LlmExecutor::getNumCacheBuffers() const {
    DCHECK_EQ(kCacheCount, getNumInputsFor(IOKind::KVCache));
    return kCacheCount;
}

std::vector<char*> LlmExecutor::getCacheBuffers() {
    const size_t numCacheInputs = getNumInputsFor(IOKind::KVCache);
    std::vector<char*> cacheBuffers(numCacheInputs);
    for (size_t i = 0; i < numCacheInputs; i++) {
        cacheBuffers[i] = reinterpret_cast<char*>(this->getInputBuffer(IOKind::KVCache, i));
    }
    return cacheBuffers;
}

void LlmExecutor::getCacheBuffersWithSize(std::vector<char*>& cacheBuffers,
                                          size_t& nBytesPerCache) {
    cacheBuffers = getCacheBuffers();
    nBytesPerCache = this->getModelInputSizeBytes(IOKind::KVCache, 0);
}

void LlmExecutor::getCacheBuffersWithSizes(std::vector<char*>& cacheBuffers,
                                           std::vector<size_t>& nBytesPerCache) {
    cacheBuffers = getCacheBuffers();
    const auto numCaches = getNumInputsFor(IOKind::KVCache);
    nBytesPerCache.resize(numCaches);
    for (size_t i = 0; i < numCaches; i++) {
        nBytesPerCache[i] = getModelInputSizeBytes(IOKind::KVCache, i);
    }
}

std::vector<size_t> LlmExecutor::getNonPaddedCacheBuffersSizes(const size_t numTokens) {
    std::vector<size_t> nBytesPerCache;
    const auto numCaches = getNumInputsFor(IOKind::KVCache);
    nBytesPerCache.resize(numCaches);
    const auto cacheSize = getCacheLength();
    const auto validNumTokens = std::min(numTokens, getNumCachedTokens());

    auto getExpectedNonPaddedSize = [=](const size_t cacheIdx) {
        // KV Cache shape: (1, num_kv_heads, cache_size, head_dim)
        return getCacheNumRows(cacheIdx) * validNumTokens * getCacheStrideSize();
    };

    for (size_t i = 0; i < numCaches; i++) {
        const size_t cacheNumBytes = getModelInputSizeBytes(IOKind::KVCache, i);
        DCHECK_EQ(cacheNumBytes % cacheSize, 0);
        nBytesPerCache[i] = cacheNumBytes / cacheSize * validNumTokens;
        DCHECK_EQ(nBytesPerCache[i], getExpectedNonPaddedSize(i));
    }
    return nBytesPerCache;
}

size_t LlmExecutor::saveNonPaddedCacheBuffers(const size_t numTokens,
                                              const std::vector<char*>& cacheBuffers) {
    const auto validNumTokens = std::min(numTokens, getNumCachedTokens());
    const auto paddedCacheBuffers = getCacheBuffers();
    const auto expectedNumBytes = getNonPaddedCacheBuffersSizes(numTokens);
    const auto numCaches = getNumInputsFor(IOKind::KVCache);
    const auto modelCacheSize = getCacheLength();

    const size_t tokenPitch = getCacheStrideSize();
    const size_t srcRowPitch = tokenPitch * modelCacheSize;
    const size_t dstRowPitch = tokenPitch * validNumTokens;
    const size_t copySize = dstRowPitch;

    DCHECK_GE(modelCacheSize, getNumCachedTokens());
    const size_t padOffset = (modelCacheSize - getNumCachedTokens()) * tokenPitch;

    for (size_t i = 0; i < numCaches; i++) {
        char* src = paddedCacheBuffers[i];
        char* dst = cacheBuffers[i];
        const auto numRows = getCacheNumRows(i);
        for (size_t rowIdx = 0; rowIdx < numRows; rowIdx++) {
            const size_t srcOffset = srcRowPitch * rowIdx + padOffset;
            const size_t dstOffset = dstRowPitch * rowIdx;
            std::memcpy(dst + dstOffset, src + srcOffset, copySize);
        }
    }
    return validNumTokens;
}

void LlmExecutor::restoreCaches(const size_t numTokens,
                                const std::vector<std::string_view>& cacheBuffers) {
    const size_t numCacheInputs = getNumInputsFor(IOKind::KVCache);
    const size_t numGivenCacheBuffers = cacheBuffers.size();
    if (numCacheInputs != numGivenCacheBuffers && numGivenCacheBuffers > 1) {
        LOG(ERROR) << "KV cache restore: the number of buffers provided (" << numGivenCacheBuffers
                   << ") does not match the number of KV cache inputs (" << numCacheInputs << ").";
        return;
    }

    resetTokenIndex(numTokens);

    // Get destination KV cache buffer to write to
    std::vector<char*> dstCacheBuffers;
    std::vector<size_t> dstCacheSizeBytes;
    getCacheBuffersWithSizes(dstCacheBuffers, dstCacheSizeBytes);
    DCHECK_EQ(dstCacheBuffers.size(), dstCacheSizeBytes.size());

    const bool requireAutoPadding = [&]() -> bool {
        if (numCacheInputs == numGivenCacheBuffers) {
            return cacheBuffers[0].size() < dstCacheSizeBytes[0]; // No need to split
        }
        const size_t totalCacheBufferSize = llm_helper::reduce_sum(dstCacheSizeBytes);
        DCHECK_EQ(cacheBuffers.size(), 1);
        DCHECK_LE(cacheBuffers[0].size(), totalCacheBufferSize);
        return cacheBuffers[0].size() < totalCacheBufferSize;
    }();

    auto getCacheCopySizeBytes = [&](const size_t cacheIdx) -> size_t {
        if (requireAutoPadding) {
            // KV Cache shape: (1, num_kv_heads, cache_size, head_dim)
            return getCacheNumRows(cacheIdx) * numTokens * getCacheStrideSize();
        }
        return dstCacheSizeBytes[cacheIdx];
    };

    // Get the source cache buffers to restore from, and split them where necessary.
    const std::vector<std::string_view> srcCacheBuffers = [&] {
        if (numCacheInputs == numGivenCacheBuffers) {
            return cacheBuffers; // No need to split
        }
        // Split the single concatenated cache buffer into per input
        DCHECK_EQ(numGivenCacheBuffers, 1);
        const auto& concatedCacheBuffer = cacheBuffers[0];
        const auto concatedCacheBufferSize = concatedCacheBuffer.size();

        std::vector<std::string_view> splitCacheBuffers;
        splitCacheBuffers.reserve(numCacheInputs);
        size_t offset = 0;
        for (size_t cacheIdx = 0; cacheIdx < numCacheInputs; cacheIdx++) {
            const size_t copySize = getCacheCopySizeBytes(cacheIdx);
            DCHECK_LE(offset + copySize, concatedCacheBufferSize);
            splitCacheBuffers.emplace_back(concatedCacheBuffer.data() + offset, copySize);
            offset += copySize;
        }
        CHECK_EQ(offset, concatedCacheBufferSize)
            << "Failed to split the concatenated cache buffer (size=" << concatedCacheBufferSize
            << ") into " << numCacheInputs << " per-input chunks.";
        LOG(DEBUG) << "restoreCaches split the concatenated cache buffers.";
        return splitCacheBuffers;
    }();

    DCHECK_EQ(srcCacheBuffers.size(), dstCacheBuffers.size());

    LOG(DEBUG) << "restoreCaches requires auto padding: " << (requireAutoPadding ? "yes" : "no");

    auto verifyCacheBufferSize = [&](const size_t idx) {
        const auto providedCacheSizeBytes = srcCacheBuffers[idx].size();
        const auto requiredCacheSizeBytes = dstCacheSizeBytes[idx];
        CHECK_LE(providedCacheSizeBytes, requiredCacheSizeBytes)
            << "Provided cache buffer size is larger than the required size.";
        if (!requireAutoPadding) {
            CHECK_EQ(providedCacheSizeBytes, requiredCacheSizeBytes)
                << "The provided cache buffer size (index=" << idx
                << ") does not match the target cache input size";
        } else {
            const auto cacheSize = getCacheLength();
            DCHECK_EQ(requiredCacheSizeBytes % cacheSize, 0);

            // 1 x num_kv_head x cache_size x head_dim / cache_size = num_kv_head x head_dim
            const size_t cacheSizeDivisor = requiredCacheSizeBytes / cacheSize;

            CHECK_EQ(providedCacheSizeBytes % cacheSizeDivisor, 0)
                << "Invalid size for the provided cache buffer (index=" << idx
                << "): " << providedCacheSizeBytes;

            const size_t inferredNumCachedTokens = providedCacheSizeBytes / cacheSizeDivisor;
            CHECK_EQ(inferredNumCachedTokens, numTokens)
                << "The provided cache buffer (index=" << idx
                << ") does not fit the provided num cached tokens.";
        }
    };

    auto restoreOneCache = [&](const size_t cacheIdx) {
        verifyCacheBufferSize(cacheIdx);
        std::memcpy(dstCacheBuffers[cacheIdx], srcCacheBuffers[cacheIdx].data(),
                    srcCacheBuffers[cacheIdx].size());
    };

    ThreadPool threadPool(ThreadSpawn::Exp); // Experiment with other thread spawn policies
    for (size_t i = 0; i < numCacheInputs; i++)
        threadPool.push(restoreOneCache, i);
    threadPool.joinAll();

    if (requireAutoPadding) {
        reconstructCaches(numTokens, getCacheLength());
    }
}

void LlmExecutor::rollbackCache(const size_t tokenCount) {
    if (tokenCount == 0) {
        return; // do nothing
    }
    DLOG_FUNC_LATENCY(ms)

    // View cache buffer of shape [..., mCacheLength, ...] as:
    //   [numRows, (mCacheLength, strideSizeBytes)]
    //    <----->  <----------------------------->
    //      row                   col

    const size_t strideSizeBytes = getCacheStrideSize();
    const size_t rowSize = mCacheLength * strideSizeBytes;

    // numCachedTokens excludes padded tokens
    const auto numCachedTokensWithPad = getNumCachedTokens() + getRightPadding();
    const size_t firstNonEmptyIdx = mCacheLength - std::min(mCacheLength, numCachedTokensWithPad);

    auto cacheBuffers = getCacheBuffers();

    // Shift right and truncate tokenCount, then fill left with zeros
    size_t cacheCounter = 0;
    for (auto cacheBuffer : cacheBuffers) {
        const size_t numRows = getCacheNumRows(cacheCounter++);
        for (size_t rowIdx = 0; rowIdx < numRows; rowIdx++) {
            auto cacheBufRow = cacheBuffer + rowIdx * rowSize; // Pointer pointing to start of row
            // Copy from back until srcOffset reaches empty segment in the cache
            for (size_t tokenIdx = mCacheLength - 1; tokenIdx >= firstNonEmptyIdx + tokenCount;
                 tokenIdx--) {
                const size_t dstOffset = tokenIdx * strideSizeBytes;
                const size_t srcOffset = (tokenIdx - tokenCount) * strideSizeBytes;
                std::memcpy(cacheBufRow + dstOffset, cacheBufRow + srcOffset, strideSizeBytes);
            }
            const size_t offset = firstNonEmptyIdx * strideSizeBytes;
            const size_t zeroCount = tokenCount * strideSizeBytes;
            std::memset(cacheBufRow + offset, 0, zeroCount);
        }
    }
}

void LlmExecutor::rollbackTreeCache(const std::vector<size_t>& acceptedIndices,
                                    size_t baseTokenSize) {
    size_t firstNonSkipIdx = 0;
    for (const size_t tokenIdx : acceptedIndices) {
        if (tokenIdx == firstNonSkipIdx) {
            firstNonSkipIdx++;
        } else {
            break;
        }
    }
    if (firstNonSkipIdx == acceptedIndices.size()) {
        return; // do nothing
    }

    if (baseTokenSize == 0)
        baseTokenSize = mModelTokenSize;

    // View cache buffer of shape [..., mCacheLength, ...] as:
    //   [numRows, (mCacheLength, strideSizeBytes)]
    //    <----->  <----------------------------->
    //      row                   col

    const size_t strideSizeBytes = getCacheStrideSize();
    const size_t rowSize = mCacheLength * strideSizeBytes;

    auto cacheBuffers = getCacheBuffers();

    size_t cacheCounter = 0;
    for (auto cacheBuffer : cacheBuffers) {
        const size_t numRows = getCacheNumRows(cacheCounter++);
        for (size_t rowIdx = 0; rowIdx < numRows; rowIdx++) {
            auto cacheBufRow = cacheBuffer + rowIdx * rowSize; // Pointer pointing to start of row
            size_t dstTokenIdx = mCacheLength - baseTokenSize + firstNonSkipIdx;
            // size_t dstTokenIdx = mCacheLength - mModelTokenSize + firstNonSkipIdx;
            for (size_t i = firstNonSkipIdx; i < acceptedIndices.size(); i++) {
                size_t tokenIdx = acceptedIndices[i];
                const size_t dstOffset = dstTokenIdx * strideSizeBytes;
                const size_t srcOffset =
                    (mCacheLength - baseTokenSize + tokenIdx) * strideSizeBytes;
                std::memcpy(cacheBufRow + dstOffset, cacheBufRow + srcOffset, strideSizeBytes);
                dstTokenIdx += 1;
            }
        }
    }
    return;
}

void LlmExecutor::setTreeAttn(const std::vector<std::vector<int>>& mask,
                              const std::vector<size_t>& positions, const size_t modelTokenSize,
                              const size_t tokenOffset) {
    mTreePositions = positions;
    mTokenSizeOffset = tokenOffset;
    mMaskBuilder->setTreeMask(mask, modelTokenSize);
}

void LlmExecutor::resetTreeAttn() {
    mTreePositions.clear();
    mMaskBuilder->reset();
    mTokenSizeOffset = 0;
}

//===------------------===//
// General Cache Eviction //
//===------------------===//

std::vector<std::string_view> LlmExecutor::getAttnWeightsBuffers() const {
    auto getAttnWgtDim1 = [this](const size_t pos) {
        uint32_t attnWgtShape[kDimensionSize];
        getRuntimeOutputShape(getOutputIndex(IOKind::Attention, pos), attnWgtShape);
        return attnWgtShape[1];
    };
    const auto numAttnOutputs = getNumOutputsFor(IOKind::Attention);
    if (!isCacheEvictionGraph()) {
        DCHECK_EQ(numAttnOutputs, 0);
        return {};
    }
    if (numAttnOutputs == 0) {
        LOG(WARN) << "Expected to use cache eviction graph but has no attention outputs.";
    }
    std::vector<std::string_view> attnWeightsBuffers(numAttnOutputs);
    for (size_t i = 0; i < numAttnOutputs; i++) {
        const auto& attnOutput = getOutput(getOutputIndex(IOKind::Attention, i));
        attnWeightsBuffers[i] =
            std::string_view(reinterpret_cast<char*>(attnOutput.buffer), attnOutput.usedSizeBytes);
        const auto attnWgtDim1 = getAttnWgtDim1(i);
        CHECK(attnWgtDim1 == kNumHead || attnWgtDim1 == 1);
    }
    return attnWeightsBuffers;
}

LlmExecutor::ShapeType LlmExecutor::getFirstAttnWeightsShape() const {
    ShapeType shape;
    if (hasOutput(IOKind::Attention)) {
        getRuntimeOutputShape(getOutputIndex(IOKind::Attention, 0), shape.data());
    }
    return shape;
}

void LlmExecutor::initCacheEviction() {
    const auto& cacheEvictionMode = kCacheEvictionOptions.mode;
    if (cacheEvictionMode == CacheEvictionMode::None) {
        return;
    }

    // Prepare KV Cache infos
    std::vector<llm_helper::KVCacheInfo> kvCacheInfos;
    kvCacheInfos.reserve(kCacheCount);
    const auto& cacheBuffers = getCacheBuffers();
    for (size_t i = 0; i < kCacheCount; i++) {
        kvCacheInfos.push_back({cacheBuffers[i], mCacheShapes[i], kCacheTypeSize});
    }

    // Create Cache Eviction Agent
    const auto defaultShrinkMode = llm_helper::CacheShrinkMode::Adaptive;
    if (cacheEvictionMode == CacheEvictionMode::LocalSnapKV) {
        mCacheEvictionAgent = std::make_unique<llm_helper::LocalSnapKVAgent>(
            kvCacheInfos, kCacheEvictionOptions.snapkvAttnSinkSize, kNumHead,
            kCacheEvictionOptions.useIngraphAttnReduce, defaultShrinkMode);
    } else if (cacheEvictionMode == CacheEvictionMode::GlobalSnapKV) {
        const auto defaultShrinkMode = llm_helper::CacheShrinkMode::Right;
        const auto& windowSize = kCacheEvictionOptions.snapkvWindowSize;
        const auto& kernelSize = kCacheEvictionOptions.snapkvKernelSize;
        const auto& maxCapacity = kCacheEvictionOptions.snapkvMaxCapacity;
        const auto& poolingMethod = kCacheEvictionOptions.snapkvPooling;

        mCacheEvictionAgent = std::make_unique<llm_helper::GlobalSnapKVAgent>(
            kvCacheInfos, windowSize, kernelSize, maxCapacity, poolingMethod, defaultShrinkMode);
    }
}

bool LlmExecutor::isEnabledCacheEviction() const {
    const auto enabledCacheEviction = kCacheEvictionOptions.mode != CacheEvictionMode::None;
    DCHECK(!enabledCacheEviction || kAttnOutputCount > 0)
        << "Cache eviction requires attention outputs.";
    return enabledCacheEviction;
}

bool LlmExecutor::isCacheEvictionGraph() const {
    if (!isEnabledCacheEviction()) {
        return false;
    }
    if (kCacheEvictionOptions.evictPromptCacheOnly && getModelTokenSize() == 1) {
        return false;
    }
    return true;
}

int LlmExecutor::runCacheEviction(const bool isCacheUpdated) {
    if (!isCacheEvictionGraph()) {
        return 0;
    }
    DCHECK(mCacheEvictionAgent);
    if (isCacheUpdated != mCacheEvictionAgent->requiresUpdatedKVCache()) {
        return 0;
    }
    mCacheEvictionAgent->updateCacheBuffers(getCacheBuffers());
    mCacheEvictionAgent->updateAttnWeights(getAttnWeightsBuffers(), getFirstAttnWeightsShape());
    mCacheEvictionAgent->updateContext(getModelTokenSize(), getCacheLength(), getNumCachedTokens(),
                                       getLeftPadding(), getRightPadding());

    llm_helper::CacheEvictionStatus evictStatus;
    if (kCacheEvictionOptions.mode == CacheEvictionMode::GlobalSnapKV) {
        // Only trigger GlobalSnapKV when the model has just consumed the entire prompt
        CHECK_GT(mNumPromptTokensHint, 0)
            << "GlobalSnapKV requires num prompt tokens hint to be set.";
        // After cache eviction has happened, num cached tokens will be < token index.
        if (mNumCachedTokens == mNumPromptTokensHint && mNumCachedTokens == getTokenIndex()) {
            evictStatus = mCacheEvictionAgent->run();
        }
    } else {
        evictStatus = mCacheEvictionAgent->run();
    }
    const auto [numEvictedTokens, overallCacheShift] = evictStatus;
    DCHECK_LE(numEvictedTokens, mNumCachedTokens)
        << "Evicted more than the number of cached tokens.";
    mNumCachedTokens -= numEvictedTokens;
    return overallCacheShift - numEvictedTokens; // Convert to right-flushed cache format
}

} // namespace mtk
