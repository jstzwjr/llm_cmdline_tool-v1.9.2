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