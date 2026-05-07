#include "llm_runtime/llava_mllm_runtime.h"

#include "backend/backend.h"
#include "common/dump.h"
#include "common/logging.h"
#include "common/thread_pool.h"
#include "common/timer.h"
#include "embedding_producer.h"
#include "executor/executor_factory.h"
#include "executor/lm_head_executor.h"
#include "executor/shared_weights.h"
#include "image_utils/image_transform.h"
#include "llm_helper/include/rotary_embedding.h"
#include "llm_helper/include/token_embedding.h"
#include "llm_helper/include/utils.h"
#include "llm_runtime/macros.h"
#include "llm_runtime/utils.h"
#include "tokenizer/tokenizer.h"

#include <string>
#include <vector>

using PatchEmbTfliteExecutor = GetExecutorClass(TFLite);
using ClipEmbDlaExecutor = GetExecutorClass(Neuron);

namespace mtk {

using TokenType = Tokenizer::TokenType;

using llm_helper::RotaryEmbeddingMasterLut;
using llm_helper::TokenEmbeddingLut;

namespace utils = llm_runtime::utils;

LlavaRuntime::LlavaRuntime() {}

LlavaRuntime::~LlavaRuntime() {}

LlavaRuntime::LlavaRuntime(const Tokenizer::TokenType imagePlaceholderToken) : MllmRuntime() {
    setImagePlaceholderToken(imagePlaceholderToken);
}

bool LlavaRuntime::initialize(const LlmModelOptions& modelOptions,
                              const LlavaRuntimeOptions& runtimeOptions,
                              const SharedWeightsHandle* preloadedSharedWeights) {
    mtk::backend::set_neuron_adapter_lib_path(runtimeOptions.npLibPath);
    if (!loadBackendLibs()) {
        return false;
    }

    mOptions = runtimeOptions;
    mClipPreprocess = runtimeOptions.clipPreprocess;

    const size_t numChunks = utils::getNumChunks(runtimeOptions);

    // External cache loading & shared weights loading
    const auto numSharedWeightsFiles = runtimeOptions.sharedWeightsFiles.size();
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
    constexpr bool duplicateSinCos = true;
    mRotEmbMasterLut = std::make_unique<RotaryEmbeddingMasterLut>(
        modelOptions.rotEmbType, modelOptions.maxTokenLength, rotEmbDim, modelOptions.rotEmbBase,
        duplicateSinCos, modelOptions.ntkScale);
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

    auto initClipExecutors = [&] {
        const auto& patchEmbFile = runtimeOptions.patchEmbFile;
        // Load CLIP Model and PatchEmb if any
        LOG(DEBUG) << "Loading CLIP DLA: " << runtimeOptions.clipFile;
        mClipExecutor = std::make_unique<ClipEmbDlaExecutor>(runtimeOptions.clipFile);
        mClipExecutor->setNumInputs(1);

        // Phase 3.1 (deepstack): Qwen3-VL encoder DLA has 7 outputs:
        //   [0]       main image embedding (always used)
        //   [1..3]    raw deepstack tap (currently emitted by shape_fixer but unused
        //             — they're already consumed inside the projector internally;
        //             a future cleanup will remove them, see report 2026-05-07.)
        //   [4..6]    projector deepstack embeddings #0/#1/#2 (LLM-dim, ready to scatter)
        // For non-deepstack models (single-output encoder), the count stays at 1.
        // TODO: read this from yaml config or detect from DLA metadata. For now hardcode.
        constexpr size_t kClipNumOutputsWithDeepstack = 7;
        constexpr size_t kClipNumOutputsLegacy = 1;
        const size_t numClipOutputs =
            runtimeOptions.clipUseDeepstack ? kClipNumOutputsWithDeepstack : kClipNumOutputsLegacy;
        mClipExecutor->setNumOutputs(numClipOutputs);
        if (runtimeOptions.clipUseDeepstack) {
            // Output indices that carry the projector deepstack embeddings (LLM-dim).
            mDeepstackOutputIndices = {4, 5, 6};
            LOG(INFO) << "[deepstack] CLIP DLA configured for "
                      << numClipOutputs << " outputs; "
                      << mDeepstackOutputIndices.size() << " deepstack embeddings will be cached.";
        } else {
            mDeepstackOutputIndices.clear();
        }

        if (!patchEmbFile.empty()) {
            mClipPatchEmbExecutor = std::make_unique<PatchEmbTfliteExecutor>(patchEmbFile);
            mClipPatchEmbExecutor->initialize();
            mClipExecutor->setModelInput(mClipPatchEmbExecutor->getOutput());
        }
        mClipExecutor->initialize();
        mClipExecutor->registerRuntimeIO();

        LOG(DEBUG) << "Initialized CLIP DLA";
    };

    // Phase 3.4 (deepstack): tell each LLM executor how many trailing ds_padded inputs
    // its DLA expects. For Qwen3-VL (merged single LLM DLA holding all 36 chunks), the
    // first executor carries all 3 ds_padded ports; for legacy multi-DLA setups, only
    // the first 3 chunks would each carry 1 ds_padded port. Until we have a robust
    // discovery path we hardcode the merged-DLA layout, which matches our actual
    // assets_*/ output. Must run BEFORE initExecutor so defineIOs sees the count.
    if (runtimeOptions.clipUseDeepstack && !mLlmDlaExecutors.empty()) {
        constexpr size_t kQwen3VlDeepstackPortCount = 3;
        mLlmDlaExecutors[0]->setDeepstackInputCount(kQwen3VlDeepstackPortCount);
        LOG(INFO) << "[deepstack] declared " << kQwen3VlDeepstackPortCount
                  << " trailing ds_padded inputs on LLM executor[0].";
    }

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
    threadPool.push(initClipExecutors);

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

void LlavaRuntime::release() {
    MllmRuntime::release();
    mClipPatchEmbExecutor.reset();
    mClipExecutor.reset();
}

void* LlavaRuntime::getImageEmbedding(const void* buffer, const size_t size) {
    size_t imageSizeBytes = 0;
    using namespace mtk::image_utils;
    const auto& pp = mClipPreprocess;
    cv::Mat image;
    Timer __wjr_ppTimer;
    __wjr_ppTimer.start();
    if (pp.mode == "qwen_vl") {
        image = qwen_vl_preprocess(buffer, size, imageSizeBytes,
                                   pp.imageWidth, pp.imageHeight,
                                   pp.patchSize, pp.temporalPatchSize, pp.mergeSize,
                                   pp.mean, pp.std, pp.resizeMode);
    } else {
        image = clip_preprocess(buffer, size, imageSizeBytes, kImgSize, kCropSize, kScale);
    }
    const double __wjr_pp_sec = __wjr_ppTimer.reset();
    LOG(INFO) << "[WJR] image preprocess took: " << __wjr_pp_sec*1000 << " ms";

    // Assume image is already preprocessed
    double __wjr_patch_sec = 0.0;
    if (mClipPatchEmbExecutor) {
        Timer patchEmbTimer;
        patchEmbTimer.start();
        mClipPatchEmbExecutor->runInference(image.data, imageSizeBytes);
        __wjr_patch_sec = patchEmbTimer.reset();
        LOG(INFO) << "Patch embedding takes: " << __wjr_patch_sec << "s";
    } else {
        mClipExecutor->setModelInput(image.data, imageSizeBytes);
    }

    // Dump encoder DLA input for alignment debug
    DUMP(INPUTS).fromBinary("encoder_input", image.data, imageSizeBytes);

    Timer clipDLATimer;
    clipDLATimer.start();
    mClipExecutor->runInference();
    const double __wjr_clip_sec = clipDLATimer.reset();
    LOG(INFO) << "Done CLIP dla inference in: " << __wjr_clip_sec << "s";
    // Accumulate vision-side time so callers can subtract it from end-to-end prompt
    // latency to get LLM-only prefill tok/s.
    addClipElapsedSeconds(__wjr_pp_sec + __wjr_patch_sec + __wjr_clip_sec);

    const auto clipEmbBuffer = mClipExecutor->getOutputBuffer(0);

    // Dump encoder DLA output for alignment debug
    const size_t clipOutputSize = mClipExecutor->getModelOutputSizeBytes(0);
    DUMP(INPUTS).fromBinary("encoder_output", clipEmbBuffer, clipOutputSize);

    // Phase 3.1 (deepstack): cache deepstack embeddings from the configured output
    // indices so callers (mllm_runtime::consumePrompt) can read them after this returns.
    mDeepstackEmbeddings.clear();
    mDeepstackEmbeddingSizes.clear();
    for (const auto outIdx : mDeepstackOutputIndices) {
        void* dsBuf = mClipExecutor->getOutputBuffer(outIdx);
        const size_t dsSize = mClipExecutor->getModelOutputSizeBytes(outIdx);
        mDeepstackEmbeddings.push_back(dsBuf);
        mDeepstackEmbeddingSizes.push_back(dsSize);
        DUMP(INPUTS).fromBinary("deepstack_embed_" + std::to_string(outIdx), dsBuf, dsSize);
    }
    if (!mDeepstackOutputIndices.empty()) {
        LOG(INFO) << "[deepstack] cached " << mDeepstackEmbeddings.size()
                  << " deepstack embeds; first size=" << mDeepstackEmbeddingSizes.front()
                  << " bytes";
    }

    return clipEmbBuffer;
}

size_t LlavaRuntime::getImageEmbeddingSize() const {
    return mClipExecutor->getModelOutputSizeBytes(0);
}

void* LlavaRuntime::getDeepstackEmbedding(const size_t idx) const {
    DCHECK_LT(idx, mDeepstackEmbeddings.size())
        << "Deepstack embedding index out of range: " << idx
        << " (have " << mDeepstackEmbeddings.size() << ")";
    return mDeepstackEmbeddings[idx];
}

size_t LlavaRuntime::getDeepstackEmbeddingSize(const size_t idx) const {
    DCHECK_LT(idx, mDeepstackEmbeddingSizes.size())
        << "Deepstack embedding size index out of range: " << idx
        << " (have " << mDeepstackEmbeddingSizes.size() << ")";
    return mDeepstackEmbeddingSizes[idx];
}

} // namespace mtk