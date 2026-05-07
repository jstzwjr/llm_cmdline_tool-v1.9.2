#pragma once

#include "common/logging.h"
#include "llm_runtime/llm_runtime.h"
#include "mtk_llm_options.h"
#include "mtk_llm_types.h"
#include "tokenizer/tokenizer.h"

#include <optional>
#include <string>
#include <vector>

#define LLM_API __attribute__((visibility("default")))

#define UNUSED_MODALITY(Modality)                                                    \
    void* get##Modality##Embedding(const void* buffer, const size_t size) override { \
        LOG(FATAL) << #Modality << " is not implemented.";                           \
        return nullptr;                                                              \
    }                                                                                \
    size_t get##Modality##EmbeddingSize() const override {                           \
        return 0;                                                                    \
    }

namespace mtk {

// Multi-modal LLM Runtime
class LLM_API MllmRuntime : public LlmRuntime {
public:
    MllmRuntime();

    virtual ~MllmRuntime();

    virtual void* getImageEmbedding(const void* buffer, const size_t size) = 0;

    virtual void* getAudioEmbedding(const void* buffer, const size_t size) = 0;

    virtual size_t getImageEmbeddingSize() const = 0;

    virtual size_t getAudioEmbeddingSize() const = 0;

    virtual void*
    consumePrompt(const Tokens& tokens, const std::vector<std::string_view>& mediaBuffers,
                  size_t* numPromptTokens, const LogitsKind outputKind = LogitsKind::LAST,
                  const bool useSoftToken = false);

    void setImagePlaceholderToken(const Tokenizer::TokenType token) {
        mImagePlaceholderToken = token;
    }

    void setAudioPlaceholderToken(const Tokenizer::TokenType token) {
        mAudioPlaceholderToken = token;
    }

    bool isImageSupported() const { return mImagePlaceholderToken.has_value(); }
    bool isAudioSupported() const { return mAudioPlaceholderToken.has_value(); }

    virtual bool isLeftPadAllowed() const override;

    // Phase 3.2 (deepstack): per-token mask over the EXPANDED prompt sequence.
    // 1 = image_pad token (target of deepstack scatter), 0 = other.
    // Built at start of consumePrompt(); read by run() to scatter ds_padded per batch.
    const std::vector<int8_t>& getVisualPosMasks() const { return mVisualPosMasks; }
    void clearVisualPosMasks() { mVisualPosMasks.clear(); }

    // CLIP timing accumulator. consumePrompt resets it; getImageEmbedding adds to it.
    // Used by callers to subtract encoder time from end-to-end prompt latency so that
    // "prompt mode tok/s" reflects LLM prefill speed only (not vision encoder).
    double getLastClipElapsedSeconds() const { return mLastClipElapsedSeconds; }
    void resetClipElapsedSeconds() { mLastClipElapsedSeconds = 0.0; }
    void addClipElapsedSeconds(const double sec) { mLastClipElapsedSeconds += sec; }

    // Phase 3.3 (deepstack): backend hooks subclass overrides to expose deepstack
    // embeddings cached by the vision encoder forward. Default: no deepstack.
    virtual void* getDeepstackEmbedding(const size_t idx) const { return nullptr; }
    virtual size_t getDeepstackEmbeddingSize(const size_t idx) const { return 0; }
    virtual size_t getNumDeepstackEmbeddings() const { return 0; }

    // Phase 3.3: per-batch ds_padded buffers built by prepareDeepstackBuffersForBatch.
    // Layout: mDeepstackPaddedBuffers[k] is a flat byte buffer of
    //         modelTokenSize * hiddenSizeBytes; k indexes which deepstack embed (0..N-1).
    // Lifetime: rebuilt for each batch inside consumePrompt's main loop.
    const std::vector<std::vector<uint8_t>>& getDeepstackPaddedBuffers() const {
        return mDeepstackPaddedBuffers;
    }

protected:
    std::vector<int8_t> mVisualPosMasks;
    std::vector<std::vector<uint8_t>> mDeepstackPaddedBuffers;
    // CLIP elapsed seconds across all images consumed in the most recent consumePrompt.
    double mLastClipElapsedSeconds = 0.0;

    // Phase 3.3: scatter the cached deepstack embeddings into ds_padded buffers shaped
    // [modelTokenSize, hiddenSizeBytes], with image_pad positions in the current batch
    // filled and all other positions (including left-pad zone) zero.
    void prepareDeepstackBuffersForBatch(size_t batchStart, size_t batchEnd,
                                          size_t leftPad, size_t modelTokenSize,
                                          size_t hiddenSizeBytes);

protected:
    Tokenizer::TokenType getImagePlaceholderToken() const noexcept {
        DCHECK(mImagePlaceholderToken.has_value()) << "Image placeholder token is not yet set.";
        return mImagePlaceholderToken.value_or(0);
    }

    Tokenizer::TokenType getAudioPlaceholderToken() const noexcept {
        DCHECK(mAudioPlaceholderToken.has_value()) << "Audio placeholder token is not yet set.";
        return mAudioPlaceholderToken.value_or(0);
    }

    std::vector<Tokenizer::TokenType> getAllPlaceholderTokens() const {
        std::vector<Tokenizer::TokenType> placeholderTokens;
        if (isImageSupported())
            placeholderTokens.push_back(getImagePlaceholderToken());
        if (isAudioSupported())
            placeholderTokens.push_back(getAudioPlaceholderToken());
        return placeholderTokens;
    }

private:
    std::optional<Tokenizer::TokenType> mImagePlaceholderToken;
    std::optional<Tokenizer::TokenType> mAudioPlaceholderToken;
};

} // namespace mtk