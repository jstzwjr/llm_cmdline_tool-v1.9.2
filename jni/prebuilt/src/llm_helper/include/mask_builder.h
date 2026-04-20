#pragma once

#include "mtk_llm_types.h"

#include <string>
#include <vector>

namespace mtk::llm_helper {

class MaskBuilder {
public:
    explicit MaskBuilder(const LLMType maskType, const size_t cacheLength);

    ~MaskBuilder();

    // Set merged mask buffer to build/update mask.
    MaskBuilder& setMaskBuffer(void* maskBuffer, const size_t maskSizeBytes);

    // Set attention mask buffer to build/update mask.
    MaskBuilder& setAttnMaskBuffer(void* maskBuffer, const size_t maskSizeBytes);

    // Set cache mask buffer to build/update mask.
    MaskBuilder& setCacheMaskBuffer(void* maskBuffer, const size_t maskSizeBytes);

    MaskBuilder& setCacheMaskBroadcastOption(const bool broadcastOption);

    // Build mask from scratch.
    void buildMask(const size_t modelTokenSize, const size_t numSeenToken);

    // Only set mask to true for seen tokens.
    // Will fallback to buildMask if mask is not updatable.
    void updateMask(const size_t modelTokenSize, const size_t numSeenToken, const size_t length);

    void notifyLeftPadding(const size_t padLength);

    void notifyRightPadding(const size_t padLength);

    // Mark mask as non-updatable which forces updateMask to call buildMask.
    void markMaskDirty();

    // Update the merged mask size. Use raw byte size to account for any HW alignment.
    void updateMaskSize(const size_t sizeBytes);

    // Update the model cache size. The mask will match the current cache size.
    void updateCacheLength(const size_t cacheLength);

    void reset();

    // Medusa
    void setMedusaTreeMask(const std::vector<std::vector<int>>& mask);

    bool isMedusaTreeAttn() const { return !mMedusaTreeMask.empty(); }

    // Llm tree candidate
    void setTreeMask(const std::vector<std::vector<int>>& mask, const size_t modelTokenSize);

    bool isTreeAttn() const { return !mTreeMask.empty(); }

    // Folded gen batch mode
    void enterFoldedGenBatchMode(const size_t numPromptTokens);

    bool isFoldedGenBatchMode() const { return mGenBatchNumPromptTokens != 0; }

    void fillPadingForTreeMask(std::vector<std::vector<int>>& mask, const size_t modelTokenSize);

private:
    template <typename MaskType>
    void buildMask(const size_t modelTokenSize, const size_t numSeenToken);

    template <typename MaskType>
    void updateMask(const size_t modelTokenSize, const size_t numSeenToken, const size_t length);

    // Adjust mask for padded input, and returns whether mask is modified for padding.
    // Used by buildMask/updateMask.
    template <typename MaskType>
    bool adjustMaskForPadding(const size_t modelTokenSize);

private:
    // Merged mask
    void* mMaskBuffer = nullptr;
    size_t mMaskSizeBytes = 0;

    // Attention mask
    void* mAttnMaskBuffer = nullptr;
    size_t mAttnMaskSizeBytes = 0;

    // Cache mask
    void* mCacheMaskBuffer = nullptr;
    size_t mCacheMaskSizeBytes = 0;
    bool mCacheMaskUseBroadcast = false;

    const LLMType kMaskType;
    const size_t kMaskTypeSize;
    size_t mCacheLength;

    // Set by notifyLeftPadding/notifyRightPadding. Reset by adjustMaskForPadding.
    size_t mLeftPadLength = 0;
    size_t mRightPadLength = 0;

    // Medusa
    std::vector<std::vector<int>> mMedusaTreeMask;

    // Llm tree candidate
    std::vector<std::vector<int>> mTreeMask;

    bool mIsMaskUpdatable = false;

    // Folded gen batch mode. Enabling will interpret token size as batch size. E.g. 8t -> 8b1t.
    // NOTE: It cannot be disabled once enabled until being reset, so chat mode is not supported.
    size_t mGenBatchNumPromptTokens = 0; // Non-zero means folded gen batch mode is enabled.
};

} // namespace mtk::llm_helper