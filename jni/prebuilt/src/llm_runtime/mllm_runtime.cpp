#include "llm_runtime/mllm_runtime.h"

#include "common/file_mem_mapper.h"
#include "embedding_producer.h"
#include "executor/llm_executor.h"
#include "common/timer.h"
#include "llm_runtime/macros.h"
#include "llm_runtime/utils.h"

#include <algorithm>
#include <cstring>

namespace mtk {

namespace utils = llm_runtime::utils;

MllmRuntime::MllmRuntime() {}

MllmRuntime::~MllmRuntime() {}

void* MllmRuntime::consumePrompt(const Tokens& tokens,
                                 const std::vector<std::string_view>& mediaBuffers,
                                 size_t* numPromptTokens, const LogitsKind outputKind,
                                 const bool useSoftToken) {
    // Get target consumer buffer
    const auto firstExecutor = getFirstLlmExecutor();
    const auto targetBuffer = firstExecutor->getInputBuffer();
    const auto targetSize = firstExecutor->getModelInputSizeBytes();

    // Prepare information for embedding producers
    const auto singleEmbSize = mTokenEmbLut->getEmbSizeBytes();

    const size_t imageTokenSize = getImageEmbeddingSize() / singleEmbSize;
    const size_t audioTokenSize = getAudioEmbeddingSize() / singleEmbSize;

    auto isImageToken = [this, &tokens](const auto start, const auto end) {
        return isImageSupported() && (tokens[start] == getImagePlaceholderToken());
    };

    auto isAudioToken = [this, &tokens](const auto start, const auto end) {
        return isAudioSupported() && (tokens[start] == getAudioPlaceholderToken());
    };

    auto loadImgEmb = [this](const std::string_view& imageBuffer) {
        return getImageEmbedding(imageBuffer.data(), imageBuffer.size());
    };

    auto loadAudioEmb = [this](const std::string_view& audioBuffer) {
        return getAudioEmbedding(audioBuffer.data(), audioBuffer.size());
    };

    // Initialize the embedding producers
    const auto subtokenIntervals = utils::partitionTokens(tokens, getAllPlaceholderTokens(), true);
    const auto numPromptSections = subtokenIntervals.size();

    std::vector<std::unique_ptr<mtk::EmbeddingProducer>> embProducerQueue;
    embProducerQueue.reserve(numPromptSections);

    *numPromptTokens = 0; // Reset

    // Phase 3.2 (deepstack): build visual_pos_masks over the EXPANDED prompt sequence.
    // image_pad in the original tokens expands to imageTokenSize positions (e.g. 400 for
    // Qwen3-VL 640x640); we mark all those expanded positions as 1.
    mVisualPosMasks.clear();
    // Reset CLIP elapsed accumulator. getImageEmbedding callbacks below will add to it.
    resetClipElapsedSeconds();

    size_t mediaIdx = 0;
    for (const auto& [start, end] : subtokenIntervals) {
        std::unique_ptr<mtk::EmbeddingProducer> curEmbProducer;
        if (isImageToken(start, end)) {
            // Image token
            CHECK_LT(mediaIdx, mediaBuffers.size())
                << "Detected more image/audio tokens than the number of given images/audios.";
            const size_t tokenSize = useSoftToken ? (end - start) : imageTokenSize;
            curEmbProducer = std::make_unique<mtk::CallbackEmbeddingProducer<std::string_view>>(
                mediaBuffers[mediaIdx++], tokenSize, loadImgEmb, singleEmbSize);
            *numPromptTokens += tokenSize;
            // Phase 3.2: image_pad expands to tokenSize positions, all marked as deepstack scatter targets.
            mVisualPosMasks.insert(mVisualPosMasks.end(), tokenSize, 1);
        } else if (isAudioToken(start, end)) {
            // Audio token
            CHECK_LT(mediaIdx, mediaBuffers.size())
                << "Detected more image/audio tokens than the number of given image/audios.";
            const size_t tokenSize = useSoftToken ? (end - start) : audioTokenSize;
            curEmbProducer = std::make_unique<mtk::CallbackEmbeddingProducer<std::string_view>>(
                mediaBuffers[mediaIdx++], tokenSize, loadAudioEmb, singleEmbSize);
            *numPromptTokens += tokenSize;
            // Phase 3.2: audio is NOT a deepstack target.
            mVisualPosMasks.insert(mVisualPosMasks.end(), tokenSize, 0);
        } else {
            // Text token
            const auto subTokens = std::vector(tokens.begin() + start, tokens.begin() + end);
            curEmbProducer = std::make_unique<mtk::TextEmbeddingProducer>(
                subTokens, mTokenEmbLut.get(), singleEmbSize);
            *numPromptTokens += subTokens.size();
            // Phase 3.2: text is not a deepstack target.
            mVisualPosMasks.insert(mVisualPosMasks.end(), subTokens.size(), 0);
        }
        DCHECK(!curEmbProducer->isEmpty());
        curEmbProducer->setConsumer(targetBuffer, targetSize);
        embProducerQueue.emplace_back(std::move(curEmbProducer));
    }
    DCHECK_EQ(mVisualPosMasks.size(), *numPromptTokens)
        << "visual_pos_masks size must equal the expanded prompt length";
    LOG(INFO) << "[deepstack] visual_pos_masks built: total=" << mVisualPosMasks.size()
              << " image_pad_count="
              << std::count(mVisualPosMasks.begin(), mVisualPosMasks.end(), int8_t{1});
    const auto& imageTokenCount = mediaIdx; // For readability in logging
    CHECK_EQ(imageTokenCount, mediaBuffers.size())
        << "The number of image tokens in the prompt does not match then number of given images.";

    // Begin consuming the prompt chunk by chunk
    auto curEmbProdIt = embProducerQueue.begin();
    auto hasProducer = [&]() { return curEmbProdIt != embProducerQueue.end(); };
    void* logitsBuffer = nullptr;
    const auto modelTokenSize = firstExecutor->getModelTokenSize();
    const auto padSize = modelTokenSize - (*numPromptTokens % modelTokenSize);

    auto getLeftPadding = [&] {
        if (isLeftPadAllowed() && firstExecutor->getTokenIndex() == 0)
            return padSize;
        return 0UL;
    };

    Timer __wjr_llmRunTimer;
    Timer __wjr_fillTimer;
    double __wjr_llmRunTotal = 0.0;
    double __wjr_fillTotal = 0.0;
    size_t __wjr_batchIdx = 0;
    // Phase 3.3 (deepstack): tracks where the current batch starts in the EXPANDED
    // prompt sequence (= same coordinate system as mVisualPosMasks).
    size_t expandedSeqCursor = 0;
    // Phase 3.3: per-token byte size for ds_padded buffers (= LLM hidden_size * dtype_bytes).
    const auto hiddenSizeBytes = (firstExecutor->getModelInputSizeBytes()
                                  / std::max<size_t>(1, modelTokenSize)
                                  / std::max<size_t>(1, firstExecutor->getBatchSize()));
    while (hasProducer()) {
        // Fill modelTokenSize number of embeddings, or break if no embedding left to consume
        const auto leftPadSize = getLeftPadding();
        size_t demandRemain = modelTokenSize - leftPadSize;
        __wjr_fillTimer.start();
        while (demandRemain > 0 && hasProducer()) {
            const auto numProduced = (*curEmbProdIt)->produceEmbedding(demandRemain);
            DCHECK_LE(numProduced, demandRemain);
            demandRemain -= numProduced;
            if ((*curEmbProdIt)->isEmpty()) {
                ++curEmbProdIt; // Move to the next producer
            }
        }
        const double __wjr_fill = __wjr_fillTimer.reset();
        __wjr_fillTotal += __wjr_fill;
        // Only the last prompt step needs logits
        const auto logitsKind = hasProducer() ? LogitsKind::NONE : outputKind;
        const auto rightPadSize = demandRemain;

        // Phase 3.3 (deepstack): scatter ds_padded for this batch BEFORE run() so
        // Phase 3.4 can pick up the buffers and feed them to LLM chunks 0/1/2.
        const auto actuallyFilled = modelTokenSize - leftPadSize - rightPadSize;
        const auto batchStart = expandedSeqCursor;
        const auto batchEnd = batchStart + actuallyFilled;
        if (getNumDeepstackEmbeddings() > 0 && hiddenSizeBytes > 0) {
            prepareDeepstackBuffersForBatch(batchStart, batchEnd, leftPadSize,
                                             modelTokenSize, hiddenSizeBytes);
        }

        __wjr_llmRunTimer.start();
        logitsBuffer = run(nullptr, leftPadSize, rightPadSize, logitsKind);
        const double __wjr_dur = __wjr_llmRunTimer.reset();
        __wjr_llmRunTotal += __wjr_dur;
        LOG(INFO) << "[WJR] LLM batch " << __wjr_batchIdx << " run() took " << __wjr_dur*1000 << " ms (fill " << __wjr_fill*1000 << " ms)";

        expandedSeqCursor = batchEnd;
        __wjr_batchIdx++;
    }
    LOG(INFO) << "[WJR] Total LLM run() time: " << __wjr_llmRunTotal << " s over " << __wjr_batchIdx << " batches (avg " << (__wjr_llmRunTotal*1000/std::max<size_t>(1,__wjr_batchIdx)) << " ms/batch)";
    LOG(INFO) << "[WJR] Total fill time: " << __wjr_fillTotal*1000 << " ms";
    return logitsBuffer;
}

bool MllmRuntime::isLeftPadAllowed() const {
    return mtk::llm_runtime::kAllowMllmLeftPadding;
}

// Phase 3.3 (deepstack): scatter the cached deepstack embeddings into per-batch
// ds_padded buffers shaped [modelTokenSize, hiddenSizeBytes].
// Phase 3.4 (deepstack): also bind those buffers into LLM chunks 0..N-1 as their
// last input port (lazy registered once on first call; subsequent calls just refill
// the contents in-place to keep buffer pointers stable for the executor).
//
// Layout assumptions:
//   - mVisualPosMasks is over the EXPANDED prompt sequence (Phase 3.2).
//   - getDeepstackEmbedding(k) returns a flat buffer [N_image_total, hiddenSize] in bytes.
//   - For Qwen3-VL @ 640x640: N_image_total = 400, hiddenSize = 2560, dtype = fp16.
//   - The current batch covers [batchStart, batchEnd) in expanded sequence coords;
//     batch buffer slot t corresponds to abs_pos = batchStart + (t - leftPad)
//     for t >= leftPad, otherwise t is in the left-pad zone (zero).
void MllmRuntime::prepareDeepstackBuffersForBatch(const size_t batchStart, const size_t batchEnd,
                                                   const size_t leftPad,
                                                   const size_t modelTokenSize,
                                                   const size_t hiddenSizeBytes) {
    const size_t numDs = getNumDeepstackEmbeddings();
    if (numDs == 0)
        return;

    const size_t bufferBytes = modelTokenSize * hiddenSizeBytes;

    // Phase 3.4 (revised): For Qwen3-VL the LLM is a single merged DLA holding all 36
    // chunks, so all numDs ds_padded inputs live on mLlmDlaExecutors[0]. defineIOs
    // (with setDeepstackInputCount called BEFORE initialize) reserved slots at the END
    // of the input list. Use getDeepstackInputIndexes() to discover the real indices
    // (e.g. [77,78,79] for Qwen3-VL @ 640x640) instead of hardcoding 7.
    if (mLlmDlaExecutors.empty()) {
        LOG(WARN) << "[deepstack] no LLM executors; skipping ds_padded scatter";
        return;
    }
    auto& mainLlmExec = mLlmDlaExecutors[0];
    const auto dsInputIdxes = mainLlmExec->getDeepstackInputIndexes();
    if (dsInputIdxes.size() != numDs) {
        LOG(WARN) << "[deepstack] expected " << numDs << " ds_padded input slots in LLM "
                  << "executor but got " << dsInputIdxes.size()
                  << "; double-check setDeepstackInputCount() was called before init.";
        return;
    }

    const bool needAlloc = (mDeepstackPaddedBuffers.size() != numDs)
                           || (numDs > 0 && mDeepstackPaddedBuffers[0].size() != bufferBytes);
    if (needAlloc) {
        mDeepstackPaddedBuffers.assign(numDs, std::vector<uint8_t>(bufferBytes, 0));
        for (size_t k = 0; k < numDs; ++k) {
            mainLlmExec->setModelInput(mDeepstackPaddedBuffers[k].data(), bufferBytes,
                                       dsInputIdxes[k]);
        }
        mainLlmExec->registerRuntimeIO();
        LOG(INFO) << "[deepstack] registered " << numDs << " ds_padded buffers ("
                  << bufferBytes << " bytes each) into LLM at input indices ["
                  << dsInputIdxes.front() << ".." << dsInputIdxes.back() << "]";
    } else {
        // Reuse existing buffers; just zero-fill in-place to preserve pointer stability.
        for (auto& buf : mDeepstackPaddedBuffers) {
            std::fill(buf.begin(), buf.end(), uint8_t{0});
        }
    }

    // Phase 3.3: count how many image_pad positions came BEFORE this batch.
    size_t consumed = 0;
    for (size_t i = 0; i < batchStart && i < mVisualPosMasks.size(); ++i) {
        if (mVisualPosMasks[i] == int8_t{1})
            consumed++;
    }
    const size_t consumedBefore = consumed;

    for (size_t t = 0; t < modelTokenSize; ++t) {
        if (t < leftPad)
            continue; // zero-padded zone
        const size_t absPos = batchStart + (t - leftPad);
        if (absPos >= mVisualPosMasks.size())
            break;
        if (mVisualPosMasks[absPos] != int8_t{1})
            continue; // non-image position stays zero

        for (size_t k = 0; k < numDs; ++k) {
            void* srcBufRaw = getDeepstackEmbedding(k);
            const size_t srcSizeBytes = getDeepstackEmbeddingSize(k);
            if (srcBufRaw == nullptr || srcSizeBytes == 0)
                continue;
            const auto srcBufBytes = static_cast<const uint8_t*>(srcBufRaw);
            const size_t srcOffsetBytes = consumed * hiddenSizeBytes;
            if (srcOffsetBytes + hiddenSizeBytes > srcSizeBytes) {
                LOG(WARN) << "[deepstack] batch " << batchStart << "-" << batchEnd
                          << " consumed=" << consumed
                          << " src_offset=" << srcOffsetBytes << " > src_size=" << srcSizeBytes
                          << " (k=" << k << "); skipping copy";
                break;
            }
            std::memcpy(mDeepstackPaddedBuffers[k].data() + t * hiddenSizeBytes,
                        srcBufBytes + srcOffsetBytes, hiddenSizeBytes);
        }
        consumed++;
    }

    LOG(INFO) << "[deepstack] batch [" << batchStart << "," << batchEnd << ") leftPad=" << leftPad
              << " consumed " << consumedBefore << "->" << consumed
              << " (per_batch_image_pad=" << (consumed - consumedBefore) << ")"
              << " bufs=" << numDs << " ds_padded_bytes=" << bufferBytes;
}

} // namespace mtk