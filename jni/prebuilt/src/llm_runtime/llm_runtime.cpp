#include "llm_runtime/llm_runtime.h"

#include "backend/backend.h"
#include "common/dump.h"
#include "common/logging.h"
#include "common/thread_pool.h"
#include "executor/executor_factory.h"
#include "executor/lm_head_executor.h"
#include "executor/shared_weights.h"
#include "llm_helper/include/rotary_embedding.h"
#include "llm_helper/include/token_embedding.h"
#include "llm_helper/include/utils.h"
#include "llm_runtime/macros.h"
#include "llm_runtime/utils.h"
#include "mtk_llm_options.h"
#include "tokenizer/tokenizer.h"

#include <regex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace mtk {

using LlmMedusaDlaExecutor = GetExecutorClass(LlmMedusa);
using TokenType = Tokenizer::TokenType;

using llm_helper::RotaryEmbeddingMasterLut;
using llm_helper::TokenEmbeddingLut;

namespace utils = llm_runtime::utils;

//===----------------------------------------------------------===
// Global Shared Weights Preloading (Independent of LlmRuntime)
//===----------------------------------------------------------===

std::unique_ptr<SharedWeightsHandle>
    LLM_API createSharedWeightsHandle(const LlmRuntimeOptions& runtimeOptions) {
    mtk::backend::set_neuron_adapter_lib_path(runtimeOptions.npLibPath);
    const size_t numChunks = utils::getNumChunks(runtimeOptions);
    const auto numSwFiles = runtimeOptions.sharedWeightsFiles.size();
    const auto numLmHeadSwFiles = static_cast<int>(runtimeOptions.useLmHeadSharedWeights);
    const size_t numDecoderSwFiles = numSwFiles > 0 ? numSwFiles - numLmHeadSwFiles : 0;

    // Error checking
    if (numDecoderSwFiles % numChunks != 0) {
        LOG(ERROR) << "Mismatch chunk count!";
        return {};
    }

    return std::make_unique<SharedWeightsHandle>(
        runtimeOptions.sharedWeightsFiles, numChunks, runtimeOptions.useLmHeadSharedWeights);
}

std::unique_ptr<SharedWeightsHandle>
    LLM_API preloadSharedWeights(const LlmRuntimeOptions& runtimeOptions) {
    auto sharedWeightsHandle = createSharedWeightsHandle(runtimeOptions);
    if (sharedWeightsHandle) {
        sharedWeightsHandle->preload();
    }
    return sharedWeightsHandle;
}

void LLM_API preloadSharedWeights(SharedWeightsHandle** sharedWeightsHandle,
                                  const LlmRuntimeOptions& runtimeOptions) {
    auto preloadedSharedWeights = preloadSharedWeights(runtimeOptions);
    *sharedWeightsHandle = preloadedSharedWeights.release();
}

void LLM_API freePreloadedSharedWeights(SharedWeightsHandle* sharedWeightsHandle) {
    delete sharedWeightsHandle;
}

void setSharedWeightsPreloadSubset(SharedWeightsHandle* sharedWeightsHandle,
                                   const std::unordered_set<size_t>& subsetIndexes,
                                   const bool repeatAllChunks) {
    CHECK(sharedWeightsHandle != nullptr);
    sharedWeightsHandle->setPreloadSubset(subsetIndexes, repeatAllChunks);
}

void LLM_API preloadSharedWeightsUnion(const std::vector<SharedWeightsHandle*>& preloadHandles,
                                       const std::vector<SharedWeightsHandle*>& refHandles) {
    SharedWeightsHandle::preloadUnion(preloadHandles, refHandles);
}

void LLM_API unloadSharedWeights(SharedWeightsHandle* sharedWeightsHandle, const size_t index) {
    CHECK(sharedWeightsHandle != nullptr);
    sharedWeightsHandle->unload(index);
}

//===---------===
//  LLM Runtime
//===---------===

LlmRuntime::LlmRuntime() : mInferenceThreadPool(2) {}

LlmRuntime::~LlmRuntime() {}

bool LlmRuntime::loadBackendLibs() {
    if constexpr (mtk::llm_runtime::kUseUsdkBackend) {
        LOG(DEBUG) << "Using NeuronUsdk (NeuronAdapter)";
    } else {
        LOG(DEBUG) << "Using Neuron Runtime";
        if (!mtk::backend::neuron_api::load_library()) {
            LOG(ERROR) << "Failed to initialize runtime library.";
            return false;
        }
    }
    return true;
}

bool LlmRuntime::initialize(const LlmModelOptions& modelOptions,
                            const LlmRuntimeOptions& runtimeOptions,
                            const SharedWeightsHandle* preloadedSharedWeights) {
    mtk::backend::set_neuron_adapter_lib_path(runtimeOptions.npLibPath);
    if (!loadBackendLibs()) {
        return false;
    }

    mOptions = runtimeOptions;

    const size_t numChunks = utils::getNumChunks(runtimeOptions);

    // External cache loading & shared weights loading
    const auto numSharedWeightsFiles = [&]() -> size_t {
        const auto numSwFiles = runtimeOptions.sharedWeightsFiles.size();
        if (numSwFiles == 0) {
            return 0;
        }
        // Exclude LM Head shared weights
        return numSwFiles - static_cast<int>(runtimeOptions.useLmHeadSharedWeights);
    }();
    const auto numCacheFiles = runtimeOptions.cacheFiles.size();
    if ((numCacheFiles > 0 && numChunks != numCacheFiles)
        || (numSharedWeightsFiles > 0 && numSharedWeightsFiles % numChunks != 0)) {
        // Mismatch chunk count
        LOG(ERROR) << "Mismatch chunk count!";
        return false;
    }

    // Use preloaded shared weights if available, otherwise create one without preload.
    const SharedWeightsHandle* sharedWeightsHandle = nullptr;
    if (preloadedSharedWeights != nullptr) {
        sharedWeightsHandle = preloadedSharedWeights;
    } else if (numSharedWeightsFiles > 0) {
        // Store the handle if it's created and owned by this llmRuntime instance
        kSharedWeightsHandle = std::make_unique<SharedWeightsHandle>(
            runtimeOptions.sharedWeightsFiles, numChunks, runtimeOptions.useLmHeadSharedWeights);
        sharedWeightsHandle = kSharedWeightsHandle.get();
    }

    // Per-chunk file getter helpers
    auto getCacheFile = [&](const size_t chunkIdx) -> FileSource {
        if (numCacheFiles > 0)
            return runtimeOptions.cacheFiles[chunkIdx];
        return {};
    };
    auto getSharedWeights = [&](const size_t chunkIdx) -> mtk::SharedWeights {
        if (sharedWeightsHandle == nullptr) {
            return {};
        }
        return sharedWeightsHandle->getSharedWeights(chunkIdx);
    };
    auto getLmHeadSharedWeights = [&]() -> mtk::SharedWeights {
        if (sharedWeightsHandle == nullptr || !sharedWeightsHandle->hasLmHead()) {
            return {};
        }
        return sharedWeightsHandle->getLmHeadSharedWeights();
    };
    auto getLoraWeightsfileMap = [&](const size_t chunkIdx) {
        std::unordered_map<LoraKey, FileSource> loraWeightsFileMap;
        if (!runtimeOptions.loraWeightsFiles.empty()) {
            for (const auto& [loraKey, loraChunkFiles] : runtimeOptions.loraWeightsFiles) {
                CHECK_EQ(loraChunkFiles.size(), numChunks)
                    << "Invalid LoRA input weights chunk count for '" << loraKey << "'";
                const auto& loraChunkFile = loraChunkFiles[chunkIdx];
                loraWeightsFileMap.emplace(loraKey, loraChunkFile);
            }
        }
        return loraWeightsFileMap;
    };

    // Get number of caches
    const size_t numCaches = 2 * modelOptions.numLayer / numChunks; // Split cache
    CHECK_EQ(modelOptions.numLayer % numChunks, 0)
        << "Requires each DLA chunk to contain equal number of layers.";
    LOG(DEBUG) << "Number of cache per dla: " << numCaches;

    // Initialize and prepare rotary embedding master lookup-table
    const size_t rotEmbDim = modelOptions.headDim;
    const size_t rotEmbLength = modelOptions.maxTokenLength;
    constexpr bool duplicateSinCos = true;
    mRotEmbMasterLut = std::make_unique<RotaryEmbeddingMasterLut>(
        modelOptions.rotEmbType, rotEmbLength, rotEmbDim, modelOptions.rotEmbBase, duplicateSinCos,
        modelOptions.ntkScale);
    mRotEmbMasterLut->generate();

    const mtk::ExecutorKind llmExecKind =
        (modelOptions.numMedusaHeads > 0) ? mtk::ExecutorKind::LlmMedusa : mtk::ExecutorKind::Llm;
    mtk::ExecutorFactory llmExecFactory(llmExecKind);

    for (size_t chunkIdx = 0; chunkIdx < numChunks; ++chunkIdx) {
        std::vector<mtk::LlmExecutor::RuntimeInfo> runtimeInfos;
        for (const auto& [modelConfig, files] : runtimeOptions.dlaFiles) {
            DCHECK_GT(files.size(), chunkIdx);
            const auto& file = files[chunkIdx];
            const auto [batchSize, tokenSize, cacheSize] =
                utils::parseModelConfig(modelConfig, modelOptions);
            runtimeInfos.push_back({file, batchSize, tokenSize, cacheSize});
            LOG(DEBUG) << "Added runtimeInfo" << "(batchSize=" << batchSize
                       << ", tokenSize=" << tokenSize << ", cacheSize=" << cacheSize
                       << "): " << file.getName();
        }

        const auto& sharedWeights = getSharedWeights(chunkIdx);
        LOG(DEBUG) << "Loading DLA " << chunkIdx;

        auto dlaExec = llmExecFactory.create<LlmExecutor>(
            runtimeInfos, modelOptions, runtimeOptions, sharedWeights, numChunks,
            mRotEmbMasterLut.get(), getLoraWeightsfileMap(chunkIdx), getCacheFile(chunkIdx));
        mLlmDlaExecutors.emplace_back(std::move(dlaExec));
    }

    mtk::ExecutorFactory neuronExecFactory(mtk::ExecutorKind::Neuron);

    if (!runtimeOptions.dlaLmHeadFile.empty()) {
        LOG(DEBUG) << "Loading and initializing Executor for LM Head.";
        mDlaLmHeadExecutor = std::make_unique<LmHeadExecutor>(
            runtimeOptions.dlaLmHeadFile, getLmHeadSharedWeights());
        mDlaLmHeadExecutor->initialize();
        mDlaLmHeadExecutor->registerRuntimeIO();
    }

    if (!runtimeOptions.dlaMedusaHeadsFile.empty()) {
        LOG(DEBUG) << "Loading and initializing Executor for Medusa Heads.";
        mDlaMedusaHeadsExecutor = neuronExecFactory.create(runtimeOptions.dlaMedusaHeadsFile);
        mDlaMedusaHeadsExecutor->setNumInputs(1);
        mDlaMedusaHeadsExecutor->setNumOutputs(1);
        mDlaMedusaHeadsExecutor->initialize();
        mDlaMedusaHeadsExecutor->registerRuntimeIO();
    }

    // Use multi-threading to speedup model loading
    ThreadPool threadPool;

    auto initExecutor = [&](const auto& dlaExec) {
        if constexpr (mtk::llm_runtime::kUseMultiThreadedLoad)
            threadPool.push(&mtk::Executor::initialize, dlaExec.get());
        else
            dlaExec->initialize();
    };

    auto initTokenEmbLut = [&] {
        // NOTE: Token embedding lookup-table type must match the model input type
        mTokenEmbLut = std::make_unique<TokenEmbeddingLut>(
            runtimeOptions.tokenEmbFile, modelOptions.modelInputType, modelOptions.hiddenSize);
        LOG(DEBUG) << "Initialized input token embedding lookup table.";
    };

    for (size_t chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
        // Initialize after reserving the input buffer so that the buffer allocator doesn't need to
        // allocate for inputs that are using an existing buffer created elsewhere.
        auto& dlaExec = mLlmDlaExecutors[chunkIdx];
        LOG(DEBUG) << "Initializing DLA " << chunkIdx;
        if (chunkIdx > 0)
            dlaExec->reserveInputBuffer(); // Prevent allocation of buffer for input 0
        initExecutor(dlaExec);
    }
    threadPool.push(initTokenEmbLut);

    // Wait for model to finish loading
    threadPool.joinAll();
    LOG(DEBUG) << "Done initializing DLAs";

    // Ensure all shared weights have been fully preloaded (if any)
    if (sharedWeightsHandle)
        sharedWeightsHandle->wait();

    // Chain the IO between the runtime chunks:
    // InputToken -> [EmbeddingLut -> DlaChunk1 -> DlaChunk2 -> ... -> DlaChunkN]-> Output
    auto getPrevChunkOutput = [&](const int chunkIdx) -> const mtk::IOBuffer& {
        DCHECK_GE(chunkIdx, 1);
        return mLlmDlaExecutors[chunkIdx - 1]->getOutput();
    };

    for (size_t chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
        // Initialize after setModelInput so that the buffer allocator doesn't need to allocate for
        // inputs that are using an existing buffer.
        auto& dlaExec = mLlmDlaExecutors[chunkIdx];
        if (chunkIdx > 0)
            dlaExec->setModelInput(getPrevChunkOutput(chunkIdx));
        dlaExec->updateModelIO(); // Ensure IO sizes are correct, esp when using prev chunk buffer
        dlaExec->registerRuntimeIO(); // Attach allocated buffers to model IO
    }
    // Link first chunk emb input to token emb lut output
    const auto& tokenEmbInput = mLlmDlaExecutors.front()->getInput();
    mTokenEmbLut->setOutput(tokenEmbInput.buffer, tokenEmbInput.sizeBytes);

    LOG(DEBUG) << "Done model chunks IO chaining";
    return true;
}

void LlmRuntime::release() {
    mLlmDlaExecutors.clear();

    mDlaLmHeadExecutor.reset();
    mDlaMedusaHeadsExecutor.reset();
    mTokenEmbLut.reset();
    mRotEmbMasterLut.reset();
    kSharedWeightsHandle.reset();
}

void* LlmRuntime::getTextEmbedding(const Tokens& inputTokens) {
    const auto firstExecutor = getFirstLlmExecutor();
    const auto modelTokenSize = firstExecutor->getModelTokenSize();
    const auto modelBatchSize = firstExecutor->getBatchSize();
    const auto maxNumInputTokens = modelTokenSize * modelBatchSize;

    // Error checking
    if (inputTokens.size() > maxNumInputTokens) {
        LOG(FATAL) << "The required input token length (" << inputTokens.size() << ") "
                   << "exceeds what the model can take in (" << maxNumInputTokens << ")";
    }

    mTokenEmbLut->lookupEmbedding(inputTokens);
    return firstExecutor->getInputBuffer();
}

Batched<void*> LlmRuntime::getTextEmbeddingBatched(const Batched<Tokens>& inputTokens,
                                                   size_t* leftPadSize, size_t* rightPadSize) {
    const auto firstExecutor = getFirstLlmExecutor();

    // Folded gen batch mode
    const auto effectiveBatchSize = getEffectiveBatchSize();
    const auto effectiveTokenSize = getEffectiveTokenSize();

    CHECK_EQ(inputTokens.size(), effectiveBatchSize)
        << "Provided batch size does not match model batch size.";
    CHECK(utils::allSameSize(inputTokens))
        << "All batches should contain the same number of tokens.";

    const auto inputTokenSize = inputTokens[0].size(); // Per batch

    // Error checking
    CHECK_LE(inputTokenSize, effectiveTokenSize)
        << "The required per-batch input token length (" << inputTokenSize << ") exceeds what the "
        << "model can take in (" << effectiveTokenSize << ")";

    // Flatten batch input tokens
    std::vector<TokenType> flattenInputTokens;

    // Handle padding
    const auto isCacheEmpty = firstExecutor->isCacheEmpty();
    const size_t padSize = effectiveTokenSize - inputTokenSize;
    constexpr TokenType padToken = 0; // By right any token should work.

    const auto inputBatchSize = inputTokens.size();
    if (inputBatchSize > 1 && padSize > 0) {
        DCHECK(leftPadSize != nullptr) << "Batched embedding lookup requires padding.";
        DCHECK(rightPadSize != nullptr) << "Batched embedding lookup requires padding.";
    }

    // The caller anticipates the possibility of padding
    const bool isPadHandled = (leftPadSize != nullptr && rightPadSize != nullptr);

    // Reset pad size
    if (isPadHandled) {
        *leftPadSize = 0;
        *rightPadSize = 0;
    }

    for (auto inputTokens : inputTokens) {
        // Use left-padding if possible as it has lower overhead than right-padding.
        // Right-padding involves cache shifting (for non-ring buffer) which incurs additional
        // overhead.
        if (isPadHandled && padSize > 0) {
            if (isLeftPadAllowed() && isCacheEmpty) {
                // Pad left since the cache is empty.
                inputTokens.insert(inputTokens.begin(), padSize, padToken);
                *leftPadSize = padSize;
                LOG(DEBUG) << "Padding left by " << padSize;
            } else {
                // Pad right since left side of cache is occupied or left pad is disabled.
                inputTokens.insert(inputTokens.end(), padSize, padToken);
                *rightPadSize = padSize;
                LOG(DEBUG) << "Padding right by " << padSize;
            }
        }
        DCHECK_EQ(effectiveTokenSize, inputTokens.size());

        // Append inputTokens to flattenInputTokens
        flattenInputTokens.insert(flattenInputTokens.end(), inputTokens.begin(), inputTokens.end());
    }

    // Lookup embeddings
    mTokenEmbLut->lookupEmbedding(flattenInputTokens);

    // Get embedding buffer
    const auto inputEmbBuffer = reinterpret_cast<char*>(firstExecutor->getInputBuffer());
    const auto inputEmbSizePerBatch = firstExecutor->getModelInputSizeBytes() / effectiveBatchSize;

    Batched<void*> embBuffers(effectiveBatchSize);
    for (size_t batch = 0; batch < effectiveBatchSize; batch++) {
        embBuffers[batch] = inputEmbBuffer + inputEmbSizePerBatch * batch;
    }
    return embBuffers;
}

// Run LLM repeatedly until the entire input embeddings are consumed
void* LlmRuntime::consumeEmbeddings(const char* embBuffer, const size_t embSize,
                                    const LogitsKind outputKind) {
    const auto singleEmbSize = mTokenEmbLut->getEmbSizeBytes();
    const auto numTokens = embSize / singleEmbSize;

    // Get target consumer buffer
    const auto firstExecutor = getFirstLlmExecutor();
    const auto targetBuffer = reinterpret_cast<char*>(firstExecutor->getInputBuffer());
    const auto targetSize = firstExecutor->getModelInputSizeBytes();

    // Begin consuming the prompt embedding chunk by chunk
    size_t numTokensRemain = numTokens;
    void* logitsBuffer = nullptr;

    const auto modelTokenSize = firstExecutor->getModelTokenSize();
    const auto padSize = modelTokenSize - (numTokens % modelTokenSize);

    auto getPadding = [&]() -> std::pair<size_t, size_t> {
        const bool isLeftPadEnabled = (isLeftPadAllowed() && firstExecutor->isCacheEmpty());
        const auto leftPadSize = isLeftPadEnabled ? padSize : 0;
        const auto rightPadSize =
            (leftPadSize == 0 && numTokensRemain == modelTokenSize - padSize) ? padSize : 0;
        return {leftPadSize, rightPadSize};
    };

    // Input of model with token size T must fit T tokens worth of embeddings
    DCHECK_GE(targetSize, modelTokenSize * singleEmbSize);

    while (numTokensRemain > 0) {
        const auto [leftPadSize, rightPadSize] = getPadding();
        const size_t writeOffset = leftPadSize * singleEmbSize;
        const size_t readOffset = (numTokens - numTokensRemain) * singleEmbSize;
        const size_t numTokensCopy = modelTokenSize - leftPadSize - rightPadSize;
        const size_t copySize = numTokensCopy * singleEmbSize;
        DCHECK_LE(numTokensCopy, numTokensRemain);
        std::memcpy(targetBuffer + writeOffset, embBuffer + readOffset, copySize);
        logitsBuffer = run(nullptr, leftPadSize, rightPadSize, outputKind);
        numTokensRemain -= numTokensCopy;
    }
    return logitsBuffer;
}

// Run LLM once with input tokens
void* LlmRuntime::run(const Tokens& inputTokens, const LogitsKind outputKind) {
    // Duplicate (broadcast) to all batches
    const auto batchInputTokens = std::vector(getEffectiveBatchSize(), inputTokens);

    // Return the first batch logits
    const auto batchLogits = runBatched(batchInputTokens, outputKind);
    DCHECK_GE(batchLogits.size(), 1);
    return batchLogits[0];
}

// Run LLM once with input embeddings
void* LlmRuntime::run(const void* inputEmb, const size_t leftPadSize, const size_t rightPadSize,
                      const LogitsKind outputKind) {
    // Return the first batch logits
    const auto batchLogits = runBatched(inputEmb, leftPadSize, rightPadSize, outputKind);
    DCHECK_GE(batchLogits.size(), 1);
    return batchLogits[0];
}

// Run LLM once with input tokens
Batched<void*> LlmRuntime::runBatched(const Batched<Tokens>& batchInputTokens,
                                      const LogitsKind outputKind) {
    // Error checks
    const auto effectiveBatchSize = getEffectiveBatchSize();
    const auto effectiveTokenSize = getEffectiveTokenSize();

    CHECK_EQ(batchInputTokens.size(), effectiveBatchSize)
        << "Provided batch size does not match model batch size.";
    CHECK(utils::allSameSize(batchInputTokens))
        << "All batches should contain the same number of tokens.";

    const auto inputTokenSize = batchInputTokens[0].size(); // Per batch
    CHECK_LE(inputTokenSize, effectiveTokenSize)
        << "The required per-batch input token length (" << inputTokenSize << ") exceeds what the "
        << "model can take in (" << effectiveTokenSize << ")";

    // Flatten batch input tokens
    std::vector<TokenType> flattenInputTokens;

    // Handle padding
    const bool isLeftPadEnabled =
        isLeftPadAllowed() && getFirstLlmExecutor()->isCacheEmpty() && !isEnabledCacheEviction();
    const size_t padSize = effectiveTokenSize - inputTokenSize;
    constexpr TokenType padToken = 0; // By right any token should work.

    size_t leftPadSize = 0, rightPadSize = 0;
    for (auto inputTokens : batchInputTokens) {
        // Use left-padding if possible as it has lower overhead than right-padding.
        // Right-padding involves cache shifting (for non-ring buffer) which incurs extra overhead.
        if (padSize > 0) {
            if (isLeftPadEnabled) {
                // Pad left since the cache is fresh new.
                inputTokens.insert(inputTokens.begin(), padSize, padToken);
                leftPadSize = padSize;
                LOG(DEBUG) << "Padding left by " << padSize;
            } else {
                // Pad right since left side of cache is occupied either by loaded cache or previous
                // inference pass.
                inputTokens.insert(inputTokens.end(), padSize, padToken);
                rightPadSize = padSize;
                LOG(DEBUG) << "Padding right by " << padSize;
            }
        }
        CHECK_EQ(effectiveTokenSize, inputTokens.size());

        // Append inputTokens to flattenInputTokens
        flattenInputTokens.insert(flattenInputTokens.end(), inputTokens.begin(), inputTokens.end());
    }

    // Lookup input embeddings from input tokens
    const auto embInputBuffer = getTextEmbedding(flattenInputTokens);
    LOG(DEBUG) << "Emb Lut output buf[0] = " << reinterpret_cast<const int16_t*>(embInputBuffer)[0];

    return runBatched(embInputBuffer, leftPadSize, rightPadSize, outputKind);
}

// Run LLM once with input embeddings
Batched<void*> LlmRuntime::runBatched(const void* inputEmb, const size_t leftPadSize,
                                      const size_t rightPadSize, const LogitsKind outputKind) {
    DCHECK(leftPadSize == 0 || rightPadSize == 0)
        << "Invalid padding: Both both left and right padding are set.";

    const auto firstExecutor = getFirstLlmExecutor();
    const auto modelTokenSize = firstExecutor->getModelTokenSize();
    const auto modelBatchSize = firstExecutor->getBatchSize();
    const auto numTotalTokens = modelTokenSize * modelBatchSize;

    const auto effectiveBatchSize = getEffectiveBatchSize();

    // Set input embedding if provided and is not identical the model input embedding buffer.
    if (inputEmb != nullptr && inputEmb != firstExecutor->getInputBuffer()) {
        const auto inputEmbSize = firstExecutor->getModelInputSizeBytes();
        firstExecutor->setModelInput(inputEmb, inputEmbSize);
        firstExecutor->registerRuntimeIO();
    }

    // Set padding if needed
    for (auto& llmDlaExec : mLlmDlaExecutors) {
        if (leftPadSize > 0)
            llmDlaExec->setLeftPadding(leftPadSize);
        else if (rightPadSize > 0)
            llmDlaExec->setRightPadding(rightPadSize);
    }

    static size_t inferenceStep = 0;
    SET_DUMP_INDEX(inferenceStep++);

    // Try advance to a larger cache size if the current input tokens will overload the cache
    const auto curCacheSize = firstExecutor->getCacheLength();
    const auto minRequiredCacheSize = firstExecutor->getNumCachedTokens() + modelTokenSize;
    if (curCacheSize < minRequiredCacheSize) {
        const auto newCacheSize = advanceCacheSize();
        if (newCacheSize > curCacheSize) {
            LOG(DEBUG) << "Advanced cache size from " << curCacheSize << " to " << newCacheSize;
        } else if (!isEnabledCacheEviction()) {
            LOG(WARN) << "Failed to advance to a larger cache size. Current cache size "
                      << "(" << curCacheSize << ") is insufficient for the current inference step.";
        }
    }

    const auto numChunks = mLlmDlaExecutors.size();

    auto getLlmDlaExec = [&](const int chunkIdx) -> LlmExecutor* {
        if (chunkIdx < 0 || chunkIdx >= numChunks)
            return nullptr;
        return mLlmDlaExecutors[chunkIdx].get();
    };

    auto dispatchThread = [&](const auto llmDlaExec, void (mtk::Executor::*func)()) {
        if (llmDlaExec == nullptr)
            return;
        if constexpr (mtk::llm_runtime::kUseInferencePipelining)
            mInferenceThreadPool.push(func, llmDlaExec);
        else
            std::invoke(func, llmDlaExec);
    };

    for (size_t chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
        auto llmDlaExec = getLlmDlaExec(chunkIdx);
        auto prevLlmDlaExec = getLlmDlaExec(chunkIdx - 1);
        auto nextLlmDlaExec = getLlmDlaExec(chunkIdx + 1);

        // First chunk prologue cannot be hidden
        if (chunkIdx == 0) {
            llmDlaExec->runInferencePrologue();
        }

        dispatchThread(prevLlmDlaExec, &mtk::Executor::runInferenceEpilogue);
        dispatchThread(nextLlmDlaExec, &mtk::Executor::runInferencePrologue);
        llmDlaExec->runInference();

        mInferenceThreadPool.wait();

        // Last chunk epilogue cannot be hidden
        if (chunkIdx == numChunks - 1) {
            llmDlaExec->runInferenceEpilogue();
        }

        SET_DUMP_CHUNK_INDEX(chunkIdx);

        // Dump chunk output
        const auto chunkOutputBuffer = llmDlaExec->getOutputBuffer();
        const auto chunkOutputSize = llmDlaExec->getModelOutputSizeBytes();
        DUMP(CHUNK_OUT).fromBinary("output", chunkOutputBuffer, chunkOutputSize);

        // Dump chunk cache outputs
        if (SHOULD_DUMP(CACHE)) {
            std::vector<char*> cacheBuffers;
            size_t sizePerCache;
            llmDlaExec->getCacheBuffersWithSize(cacheBuffers, sizePerCache);
            for (size_t i = 0; i < cacheBuffers.size(); i++) {
                DUMP(CACHE).fromBinary("cache_" + std::to_string(i), cacheBuffers[i], sizePerCache);
            }
        }
    }

    // Return logits

    const auto finalExecutor = getFinalLlmExecutor();
    const auto& lmHeadExecutor = mDlaLmHeadExecutor;

    if (!lmHeadExecutor) {
        // No separated LM head, return the logits from the final executor directly.
        return getLogitsBuffer(finalExecutor, numTotalTokens, rightPadSize, outputKind);
    }

    if (outputKind == LogitsKind::NONE) {
        // Logits is not required, so no need to run lmHead.
        return std::vector<void*>(effectiveBatchSize, nullptr);
    }

    // Execute the LM head on the hidden state generated from the last chunk of decoder layers.
    const auto hiddenStateSize = finalExecutor->getModelOutputSizeBytes();
    const auto lmHeadInputSize = lmHeadExecutor->getModelInputSizeBytes();
    const auto perTokenHiddenStateSize = hiddenStateSize / numTotalTokens;
    const auto lmHeadNumTotalTokens = lmHeadInputSize / perTokenHiddenStateSize;

    // Ensure LM Head input token size is large enough to process all batches (during batch mode)
    if (effectiveBatchSize > 1) {
        CHECK_LE(hiddenStateSize, lmHeadInputSize)
            << "Batch model requires LM Head with sufficient size: Batch model output hidden size "
            << "(" << hiddenStateSize << ") > LM Head input size (" << lmHeadInputSize << ")";
    }

    // The input hidden state to LM Head is left-flushed so only the last N non-padded tokens are
    // processed, where N is the LM Head token size.
    const size_t hiddenStateTokenOffset =
        utils::max0Subtract(numTotalTokens, lmHeadNumTotalTokens + rightPadSize);

    const size_t hiddenStateOffset = hiddenStateTokenOffset * perTokenHiddenStateSize;
    DCHECK_LE(hiddenStateOffset, hiddenStateSize);
    auto hiddenStateBuffer = reinterpret_cast<char*>(finalExecutor->getOutputBuffer());
    lmHeadExecutor->runInference(hiddenStateBuffer + hiddenStateOffset, lmHeadInputSize);

    // Return logits from LM head output
    if (outputKind == LogitsKind::FULL) {
        // If the logits of all the input tokens are expected, the token size of LM-Head chunk
        // must be large enough for the total token size of the currently used chunks.
        DCHECK_LE(numTotalTokens, lmHeadNumTotalTokens);
    }
    return getLogitsBuffer(lmHeadExecutor.get(), lmHeadNumTotalTokens, rightPadSize, outputKind);
}

void LlmRuntime::reset(const bool resetCache) {
    for (auto& llmDlaExec : mLlmDlaExecutors) {
        if (resetCache) {
            // Reset cache and token index, resetTokenIndex() will be called
            llmDlaExec->initCache();
        } else {
            // Reset token index without resetting cache
            llmDlaExec->resetTokenIndex();
        }
    }
}

void LlmRuntime::rollback(const size_t count) {
    if (count == 0)
        return;
    const auto modelTokenSize = getFinalLlmExecutor()->getModelTokenSize();
    for (auto& llmDlaExec : mLlmDlaExecutors) {
        // Align token index and rollback cache
        llmDlaExec->alignInputTokens(modelTokenSize - count);
    }
}

void LlmRuntime::swapModel(const size_t tokenSize, const size_t cacheSize) {
    // Use multi-threading to speedup model swapping (if necessary)
    ThreadPool threadPool;

    auto swapModel = [&](const auto chunkIdx) {
        auto& llmDlaExec = mLlmDlaExecutors[chunkIdx];
        if (!llmDlaExec->hotSwapModel(tokenSize, cacheSize)) {
            LOG(ERROR) << "Hot swapping failed on chunk " << chunkIdx;
        }
    };
    const auto numDlaChunk = mLlmDlaExecutors.size();
    for (size_t chunkIdx = 0; chunkIdx < numDlaChunk; chunkIdx++) {
        if constexpr (!mtk::llm_runtime::kUseMultiThreadedLoad)
            swapModel(chunkIdx);
        else
            threadPool.push(swapModel, chunkIdx);
    }

    // Wait for model swapping threads to finish
    threadPool.joinAll();
}

size_t LlmRuntime::advanceCacheSize() {
    // 1. Find next larger cache size
    // 2. Call swapModel(curTokenSize, nextCacheSize)
    // 3. Return new cache size
    const auto firstExecutor = getFirstLlmExecutor();
    const auto curTokenSize = firstExecutor->getModelTokenSize();
    const auto curCacheSize = firstExecutor->getCacheLength();
    const auto nextLargerCacheSize = firstExecutor->getNextAvailCacheSize();
    if (nextLargerCacheSize > curCacheSize) {
        Timer cacheAdvanceTimer;
        cacheAdvanceTimer.start();
        swapModel(curTokenSize, nextLargerCacheSize);
        LOG(DEBUG) << "Advancing " << curTokenSize << "t model cache size from " << curCacheSize
                   << " to " << nextLargerCacheSize << " took " << cacheAdvanceTimer.reset() * 1000
                   << " ms";
    }
    return nextLargerCacheSize;
}

void LlmRuntime::enterFoldedGenBatchMode() {
    for (auto& llmDlaExec : mLlmDlaExecutors) {
        llmDlaExec->enterFoldedGenBatchMode();
    }
}

size_t LlmRuntime::getPerTokenLogitsSize() const {
    const auto finalExecutor = getFinalLlmExecutor();
    const auto modelBatchSize = finalExecutor->getBatchSize();
    const auto modelTokenSize = finalExecutor->getModelTokenSize();
    const auto numTotalTokens = modelTokenSize * modelBatchSize;
    if (!isSeparateLmHead()) {
        // Final executor output[0] is logits
        const auto logitsSize = finalExecutor->getModelOutputSizeBytes();
        return logitsSize / numTotalTokens;
    } else {
        // LmHead executor output is logits
        const auto perTokenHiddenStateSize =
            finalExecutor->getModelOutputSizeBytes() / numTotalTokens;
        const auto lmHeadNumTotalTokens =
            mDlaLmHeadExecutor->getModelInputSizeBytes() / perTokenHiddenStateSize;
        const auto logitsSize = mDlaLmHeadExecutor->getModelOutputSizeBytes();
        return logitsSize / lmHeadNumTotalTokens;
    }
}

size_t LlmRuntime::getPerTokenHiddenStatesSize() const {
    // Error checking
    if (!isSeparateLmHead()) {
        LOG(FATAL) << "Separated LM Head is necessary for calculating the size of hidden states.";
    }
    const auto finalExecutor = getFinalLlmExecutor();
    const auto modelTokenSize = finalExecutor->getModelTokenSize();
    const auto modelBatchSize = finalExecutor->getBatchSize();
    const auto hiddenStateSize = finalExecutor->getModelOutputSizeBytes();
    return hiddenStateSize / modelTokenSize / modelBatchSize;
}

size_t LlmRuntime::getTokenIndex() const {
    return getFirstLlmExecutor()->getTokenIndex();
}

bool LlmRuntime::isSeparateLmHead() const {
    return (mDlaLmHeadExecutor != nullptr);
}

void* LlmRuntime::getLastHiddenStates() {
    return getFinalLlmExecutor()->getOutputBuffer();
}

void LlmRuntime::getKVCaches(std::vector<std::vector<char*>>* kvCaches,
                             std::vector<std::vector<size_t>>* sizes) const {
    const auto numDlaChunks = mLlmDlaExecutors.size();

    kvCaches->clear();
    kvCaches->reserve(numDlaChunks);
    sizes->clear();
    sizes->reserve(numDlaChunks);

    for (auto& llmDlaExec : mLlmDlaExecutors) {
        std::vector<char*> chunkCaches;
        std::vector<size_t> chunkSizes;
        llmDlaExec->getCacheBuffersWithSizes(chunkCaches, chunkSizes);
        DCHECK_EQ(chunkCaches.size(), chunkSizes.size());
        kvCaches->push_back(std::move(chunkCaches));
        sizes->push_back(std::move(chunkSizes));
    }
}

std::vector<size_t> LlmRuntime::getNonPaddedCachesNumBytes(const size_t numTokens) const {
    std::vector<size_t> sizes;
    sizes.reserve([&] {
        size_t totalCacheBuffers = 0;
        for (auto& llmDlaExec : mLlmDlaExecutors) {
            totalCacheBuffers += llmDlaExec->getNumCacheBuffers();
        }
        return totalCacheBuffers;
    }());
    for (auto& llmDlaExec : mLlmDlaExecutors) {
        const auto chunkCacheSizes = llmDlaExec->getNonPaddedCacheBuffersSizes(numTokens);
        // Append chunkCacheSizes to sizes
        std::copy(chunkCacheSizes.begin(), chunkCacheSizes.end(), std::back_inserter(sizes));
    }
    return sizes;
}

size_t LlmRuntime::saveNonPaddedCaches(const size_t numTokens,
                                       const std::vector<char*>& dstBuffers) {
    size_t startCacheIdx = 0;
    auto getNextKVCaches = [&](const size_t count) {
        const size_t start = startCacheIdx;
        const size_t end = start + count;
        startCacheIdx = end; // Update
        DCHECK_LE(end, dstBuffers.size());
        return std::vector(dstBuffers.begin() + start, dstBuffers.begin() + end);
    };

    size_t validNumTokens = 0;
    for (auto& llmDlaExec : mLlmDlaExecutors) {
        const auto cacheCount = llmDlaExec->getNumCacheBuffers();
        validNumTokens =
            llmDlaExec->saveNonPaddedCacheBuffers(numTokens, getNextKVCaches(cacheCount));
    }
    DCHECK_LE(validNumTokens, numTokens);
    return validNumTokens;
}

void LlmRuntime::restoreKVCaches(const size_t numTokens,
                                 const std::vector<std::string_view>& kvCaches) {
    const auto numDlaChunks = mLlmDlaExecutors.size();
    CHECK_EQ(kvCaches.size() % numDlaChunks, 0)
        << "Unable to distribute the KV cache buffers to the executor chunks.";
    const auto numCachesPerChunk = kvCaches.size() / numDlaChunks;
    auto getSubsetForChunk = [&](const size_t chunkIdx) {
        const size_t start = chunkIdx * numCachesPerChunk;
        const size_t end = start + numCachesPerChunk;
        DCHECK_LE(end, kvCaches.size());
        return std::vector(kvCaches.begin() + start, kvCaches.begin() + end);
    };
    for (size_t chunkIdx = 0; chunkIdx < numDlaChunks; chunkIdx++) {
        auto& llmDlaExec = mLlmDlaExecutors[chunkIdx];
        llmDlaExec->restoreCaches(numTokens, getSubsetForChunk(chunkIdx));
    }
}

void LlmRuntime::setLlmTreeAttn(const std::vector<std::vector<int>>& mask,
                                const std::vector<size_t>& positions, const size_t modelTokenSize,
                                const size_t tokenOffset) {
    for (auto& llmDlaExec : mLlmDlaExecutors) {
        llmDlaExec->setTreeAttn(mask, positions, modelTokenSize, tokenOffset);
    }
}

void LlmRuntime::resetLlmTreeAttn() {
    for (auto& llmDlaExec : mLlmDlaExecutors) {
        llmDlaExec->resetTreeAttn();
    }
}

void LlmRuntime::llmTreeRollback(const std::vector<size_t>& acceptedIndices,
                                 const size_t relativeBeginOffset) {
    for (auto& llmDlaExec : mLlmDlaExecutors) {
        llmDlaExec->rollbackTreeCache(acceptedIndices, relativeBeginOffset);
        llmDlaExec->alignInputTokens(acceptedIndices.size(), relativeBeginOffset);
    }
}

//===-----===//
// LoRA APIs
//===-----===//

void LlmRuntime::applyLora(const LoraKey& loraKey) {
    for (auto& llmDlaExec : mLlmDlaExecutors) {
        llmDlaExec->applyLoraWeights(loraKey);
    }
}

void LlmRuntime::applyLoraFromBuffer(const std::vector<const char*>& loraWeightBuffers,
                                     const std::vector<size_t>& sizes) {
    const auto& loraInputCount = mOptions.loraInputCount; // Per chunk
    const auto& chunkCount = mLlmDlaExecutors.size();

    // Verify arguments
    CHECK_EQ(loraWeightBuffers.size(), sizes.size());
    CHECK_EQ(chunkCount * loraInputCount, loraWeightBuffers.size())
        << "The provided number of LoRA weights buffers does not match the total number of "
           "LoRA inputs";

    auto getSubsetForChunk = [loraInputCount](const size_t chunkIdx, const auto& vec) {
        const size_t start = chunkIdx * loraInputCount;
        const size_t end = start + loraInputCount;
        return std::vector(vec.begin() + start, vec.begin() + end);
    };

    // Chunk the LoRA weight buffers and feed into each DLA chunk.
    for (size_t chunkIdx = 0; chunkIdx < chunkCount; chunkIdx++) {
        const auto& loraWeightForChunk = getSubsetForChunk(chunkIdx, loraWeightBuffers);
        const auto& sizesForChunk = getSubsetForChunk(chunkIdx, sizes);
        auto& llmDlaExec = mLlmDlaExecutors[chunkIdx];
        llmDlaExec->applyLoraWeights(loraWeightForChunk, sizesForChunk);
    }
}

void LlmRuntime::removeLora() {
    for (auto& llmDlaExec : mLlmDlaExecutors) {
        llmDlaExec->removeLoraWeights();
    }
}

//===-------===//
// Medusa APIs
//===-------===//

void LlmRuntime::setMedusaTreeAttn(const std::vector<std::vector<int>>& mask,
                                   const std::vector<size_t>& positions) {
    for (auto& llmDlaExec : mLlmDlaExecutors) {
        auto llmMedusaDlaExec = static_cast<LlmMedusaDlaExecutor*>(llmDlaExec.get());
        llmMedusaDlaExec->setMedusaTreeAttn(mask, positions);
    }
}

void* LlmRuntime::runMedusaHeads(const void* hiddenState) {
    // Error checking
    CHECK(mDlaMedusaHeadsExecutor != nullptr) << "Medusa Heads is necessary for Medusa inference.";

    const auto& medusaExecutor = mDlaMedusaHeadsExecutor;

    medusaExecutor->runInference(hiddenState, medusaExecutor->getModelInputSizeBytes());
    auto logitsBuffer = medusaExecutor->getOutputBuffer();

    return logitsBuffer;
}

void LlmRuntime::medusaRollback(const std::vector<size_t>& acceptedIndices) {
    for (auto& llmDlaExec : mLlmDlaExecutors) {
        auto llmMedusaDlaExec = static_cast<LlmMedusaDlaExecutor*>(llmDlaExec.get());
        llmMedusaDlaExec->rollbackTreeCache(acceptedIndices);
        llmMedusaDlaExec->alignInputTokens(acceptedIndices.size());
    }
}

//===---------------===//
// Cacge Eviction APIs
//===---------------===//

bool LlmRuntime::isEnabledCacheEviction() const {
    return getFirstLlmExecutor()->isEnabledCacheEviction();
}

//===------------===//
// Helper Functions
//===------------===//

LlmExecutor* LlmRuntime::getFirstLlmExecutor() {
    CHECK(!mLlmDlaExecutors.empty());
    return mLlmDlaExecutors.front().get();
}

LlmExecutor* LlmRuntime::getFinalLlmExecutor() {
    CHECK(!mLlmDlaExecutors.empty());
    return mLlmDlaExecutors.back().get();
}

const LlmExecutor* LlmRuntime::getFirstLlmExecutor() const {
    CHECK(!mLlmDlaExecutors.empty());
    return mLlmDlaExecutors.front().get();
}

const LlmExecutor* LlmRuntime::getFinalLlmExecutor() const {
    CHECK(!mLlmDlaExecutors.empty());
    return mLlmDlaExecutors.back().get();
}

size_t LlmRuntime::getEffectiveBatchSize() const {
    const auto firstExecutor = getFirstLlmExecutor();
    if (firstExecutor->isFoldedGenBatchMode())
        return firstExecutor->getModelTokenSize();
    return firstExecutor->getBatchSize();
}

size_t LlmRuntime::getEffectiveTokenSize() const {
    const auto firstExecutor = getFirstLlmExecutor();
    if (firstExecutor->isFoldedGenBatchMode())
        return 1;
    return firstExecutor->getModelTokenSize();
}

void LlmRuntime::setNumPromptTokensHint(const size_t count) {
    for (auto& llmDlaExec : mLlmDlaExecutors) {
        llmDlaExec->setNumPromptTokensHint(count);
    }
}

Batched<void*> LlmRuntime::getLogitsBuffer(Executor* executor, const size_t execNumTokens,
                                           const size_t rightPadSize,
                                           const LogitsKind outputKind) const {
    auto logitsBuffer = reinterpret_cast<char*>(executor->getOutputBuffer());
    const size_t logitsSizePerToken = executor->getModelOutputSizeBytes() / execNumTokens;
    const auto effectiveTokenSize = getEffectiveTokenSize();
    const auto effectiveBatchSize = getEffectiveBatchSize();
    const size_t logitsSizePerBatch = logitsSizePerToken * effectiveTokenSize;
    DCHECK_GE(effectiveTokenSize, rightPadSize);
    // Use min() to handle the case where lmHead size < hiddenState size
    const auto numValidTokensPerBatch = std::min(effectiveTokenSize - rightPadSize, execNumTokens);
    DCHECK_GE(numValidTokensPerBatch, 1);
    size_t logitsOffset = 0;
    if (outputKind == LogitsKind::LAST) {
        logitsOffset = logitsSizePerToken * (numValidTokensPerBatch - 1);
        DCHECK_LE(logitsOffset, logitsSizePerBatch);
    }
    Batched<void*> logits(effectiveBatchSize);
    for (size_t batch = 0; batch < effectiveBatchSize; batch++) {
        auto curLogitsBuffer = logitsBuffer + logitsSizePerBatch * batch;
        logits[batch] = curLogitsBuffer + logitsOffset;
    }
    return logits;
}

bool LlmRuntime::isLeftPadAllowed() const {
    return mtk::llm_runtime::kAllowLlmLeftPadding;
}

} // namespace mtk
