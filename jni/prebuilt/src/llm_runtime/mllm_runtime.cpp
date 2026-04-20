#include "llm_runtime/mllm_runtime.h"

#include "common/file_mem_mapper.h"
#include "embedding_producer.h"
#include "executor/llm_executor.h"
#include "llm_runtime/macros.h"
#include "llm_runtime/utils.h"

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
        } else if (isAudioToken(start, end)) {
            // Audio token
            CHECK_LT(mediaIdx, mediaBuffers.size())
                << "Detected more image/audio tokens than the number of given image/audios.";
            const size_t tokenSize = useSoftToken ? (end - start) : audioTokenSize;
            curEmbProducer = std::make_unique<mtk::CallbackEmbeddingProducer<std::string_view>>(
                mediaBuffers[mediaIdx++], tokenSize, loadAudioEmb, singleEmbSize);
            *numPromptTokens += tokenSize;
        } else {
            // Text token
            const auto subTokens = std::vector(tokens.begin() + start, tokens.begin() + end);
            curEmbProducer = std::make_unique<mtk::TextEmbeddingProducer>(
                subTokens, mTokenEmbLut.get(), singleEmbSize);
            *numPromptTokens += subTokens.size();
        }
        DCHECK(!curEmbProducer->isEmpty());
        curEmbProducer->setConsumer(targetBuffer, targetSize);
        embProducerQueue.emplace_back(std::move(curEmbProducer));
    }
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

    while (hasProducer()) {
        // Fill modelTokenSize number of embeddings, or break if no embedding left to consume
        const auto leftPadSize = getLeftPadding();
        size_t demandRemain = modelTokenSize - leftPadSize;
        while (demandRemain > 0 && hasProducer()) {
            const auto numProduced = (*curEmbProdIt)->produceEmbedding(demandRemain);
            DCHECK_LE(numProduced, demandRemain);
            demandRemain -= numProduced;
            if ((*curEmbProdIt)->isEmpty()) {
                ++curEmbProdIt; // Move to the next producer
            }
        }
        // Only the last prompt step needs logits
        const auto logitsKind = hasProducer() ? LogitsKind::NONE : outputKind;
        const auto rightPadSize = demandRemain;
        logitsBuffer = run(nullptr, leftPadSize, rightPadSize, logitsKind);
    }
    return logitsBuffer;
}

bool MllmRuntime::isLeftPadAllowed() const {
    return mtk::llm_runtime::kAllowMllmLeftPadding;
}

} // namespace mtk