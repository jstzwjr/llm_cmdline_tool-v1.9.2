#include "llm_helper/include/mask_builder.h"

#include "common/logging.h"
#include "llm_helper/include/utils.h"

namespace mtk::llm_helper {

// Define mask values for different types
template <typename T>
struct MaskVal;

#define __DECL_MASK__(TYPE, TRUE_VAL, FALSE_VAL)  \
    template <>                                   \
    struct MaskVal<TYPE> {                        \
        static constexpr TYPE kTrue = TRUE_VAL;   \
        static constexpr TYPE kFalse = FALSE_VAL; \
    };

__DECL_MASK__(bool, true, false)
__DECL_MASK__(int8_t, 0, -128)
__DECL_MASK__(int16_t, 0, -32768)
__DECL_MASK__(__fp16, 0, -100)
__DECL_MASK__(float, 0, -100)
#undef __DECL_MASK__

MaskBuilder::MaskBuilder(const LLMType maskType, const size_t cacheLength)
    : kMaskType(maskType), kMaskTypeSize(getLLMTypeSize(maskType)), mCacheLength(cacheLength) {}

MaskBuilder::~MaskBuilder() {}

MaskBuilder& MaskBuilder::setMaskBuffer(void* maskBuffer, const size_t maskSizeBytes) {
    // Enable merged mask
    mMaskBuffer = maskBuffer;
    mMaskSizeBytes = maskSizeBytes;
    // Disable split mask
    mAttnMaskBuffer = nullptr;
    mCacheMaskBuffer = nullptr;
    return *this;
}

MaskBuilder& MaskBuilder::setAttnMaskBuffer(void* maskBuffer, const size_t maskSizeBytes) {
    // Enable split mask
    mAttnMaskBuffer = maskBuffer;
    mAttnMaskSizeBytes = maskSizeBytes;
    // Disable merged mask
    mMaskBuffer = nullptr;
    return *this;
}

MaskBuilder& MaskBuilder::setCacheMaskBuffer(void* maskBuffer, const size_t maskSizeBytes) {
    // Enable split mask
    mCacheMaskBuffer = maskBuffer;
    mCacheMaskSizeBytes = maskSizeBytes;
    // Disable merged mask
    mMaskBuffer = nullptr;
    return *this;
}

MaskBuilder& MaskBuilder::setCacheMaskBroadcastOption(const bool broadcastOption) {
    mCacheMaskUseBroadcast = broadcastOption;
    return *this;
}

void MaskBuilder::updateMaskSize(const size_t sizeBytes) {
    mMaskSizeBytes = sizeBytes;
}

void MaskBuilder::updateCacheLength(const size_t cacheLength) {
    mCacheLength = cacheLength;
}

void MaskBuilder::markMaskDirty() {
    mIsMaskUpdatable = false;
}

template <typename MaskType>
void MaskBuilder::buildMask(const size_t modelTokenSize, const size_t numSeenToken) {
    constexpr auto maskTrue = MaskVal<MaskType>::kTrue;
    constexpr auto maskFalse = MaskVal<MaskType>::kFalse;

    const size_t cacheMaskNumRows = mCacheMaskUseBroadcast ? 1u : modelTokenSize;
    const auto [cacheMaskBuffer, cacheMaskRowStride] = [&] {
        MaskType* cacheMaskBuffer = nullptr;
        size_t rowStride = 0; // Num vals per row
        if (mCacheMaskBuffer) {
            // Split-mask
            cacheMaskBuffer = reinterpret_cast<MaskType*>(mCacheMaskBuffer);
            rowStride = mCacheMaskSizeBytes / cacheMaskNumRows / kMaskTypeSize;
        } else {
            // Merged-mask
            cacheMaskBuffer = reinterpret_cast<MaskType*>(mMaskBuffer);
            rowStride = mMaskSizeBytes / modelTokenSize / kMaskTypeSize;
        }
        DCHECK(cacheMaskBuffer);
        DCHECK_GT(rowStride, 0);
        return std::pair{cacheMaskBuffer, rowStride};
    }();

    const size_t attnMaskNumRows = modelTokenSize;
    const auto [attnMaskBuffer, attnMaskRowStride] = [this, modelTokenSize] {
        MaskType* attnMaskBuffer = nullptr;
        size_t rowStride = 0; // Num vals per row
        if (mAttnMaskBuffer) {
            // Split-mask
            attnMaskBuffer = reinterpret_cast<MaskType*>(mAttnMaskBuffer);
            rowStride = mAttnMaskSizeBytes / modelTokenSize / kMaskTypeSize;
        } else {
            // Merged-mask
            attnMaskBuffer = reinterpret_cast<MaskType*>(mMaskBuffer) + mCacheLength;
            rowStride = mMaskSizeBytes / modelTokenSize / kMaskTypeSize;
        }
        DCHECK(attnMaskBuffer);
        DCHECK_GT(rowStride, 0);
        return std::pair{attnMaskBuffer, rowStride};
    }();

    // The first valid cache index to set mask as True
    const size_t startValidCacheIdx = mCacheLength - std::min(mCacheLength, numSeenToken);

    if (isFoldedGenBatchMode()) {
        CHECK_EQ(cacheMaskNumRows, modelTokenSize)
            << "Folded batch gen requires non-broadcast cache mask input.";
        // Token size is interpreted as batch size
        const auto& foldedBatchSize = modelTokenSize;
        DCHECK_EQ(mLeftPadLength + mRightPadLength, 0); // Padding is not allowed in this mode
        // There are modelTokenSize number of rows
        for (size_t rowIdx = 0; rowIdx < cacheMaskNumRows; rowIdx++) {
            // Set the (rectangle) input cache mask as True for prompt tokens
            auto cacheMaskRow = cacheMaskBuffer + rowIdx * cacheMaskRowStride;
            size_t i = 0; // Cache mask column write index
            while (i < startValidCacheIdx)
                cacheMaskRow[i++] = maskFalse;
            while (i < startValidCacheIdx + mGenBatchNumPromptTokens)
                cacheMaskRow[i++] = maskTrue;

            DCHECK_LE(i, mCacheLength);
            CHECK_EQ((mCacheLength - i) % foldedBatchSize, 0)
                << "Please ensure the cache size is sufficient for gen batch mode.";

            // Set the identity matrices at the right to interpret token size as batch size
            size_t batchIdx = 0;
            while (i < mCacheLength) {
                cacheMaskRow[i++] = (batchIdx == rowIdx) ? maskTrue : maskFalse;
                batchIdx = (batchIdx + 1) % foldedBatchSize;
            }

            // Set attention mask as an identity matrix
            DCHECK_EQ(cacheMaskNumRows, attnMaskNumRows); // No support cache mask broadcast
            auto attnMaskRow = attnMaskBuffer + rowIdx * attnMaskRowStride;
            DCHECK_EQ(batchIdx, 0);
            while (batchIdx < modelTokenSize) {
                const auto maskValue = (batchIdx == rowIdx) ? maskTrue : maskFalse;
                attnMaskRow[batchIdx++] = maskValue;
            }
        }
        mIsMaskUpdatable = false; // Disable mask update for folded batch
        return;
    }

    // Cache mask
    DCHECK_EQ(mCacheMaskSizeBytes % (cacheMaskNumRows * kMaskTypeSize), 0);
    for (size_t rowIdx = 0; rowIdx < cacheMaskNumRows; rowIdx++) {
        auto cacheMaskRow = cacheMaskBuffer + rowIdx * cacheMaskRowStride;
        size_t i = 0; // Cache mask column write index
        while (i < startValidCacheIdx)
            cacheMaskRow[i++] = maskFalse;
        while (i < mCacheLength)
            cacheMaskRow[i++] = maskTrue;
    }

    // Attention mask
    CHECK(!isMedusaTreeAttn() || !isTreeAttn());
    if (isMedusaTreeAttn()) {
        DCHECK_EQ(mLeftPadLength, 0)
            << "For medusa inference, tree-candidate length must align with genTokenSize.";
        DCHECK_EQ(mRightPadLength, 0)
            << "For medusa inference, tree-candidate length must align with genTokenSize.";
    }
    DCHECK_EQ(mAttnMaskSizeBytes % (modelTokenSize * kMaskTypeSize), 0);

    for (size_t rowIdx = 0; rowIdx < attnMaskNumRows; rowIdx++) {
        auto attnMaskRow = attnMaskBuffer + rowIdx * attnMaskRowStride;
        size_t i = 0; // Attention mask column write index
        if (!isMedusaTreeAttn() && !isTreeAttn()) {
            // Normal causal mask
            const size_t attnTrueCount = rowIdx + 1;
            while (i < attnTrueCount)
                attnMaskRow[i++] = maskTrue;
            while (i < modelTokenSize)
                attnMaskRow[i++] = maskFalse;
        } else if (isMedusaTreeAttn()) {
            // Medusa special mask (already saved in mMedusaTreeMask)
            for (const auto medusaMaskVal : mMedusaTreeMask[rowIdx]) {
                attnMaskRow[i++] = (medusaMaskVal == 1) ? maskTrue : maskFalse;
            }
            DCHECK_EQ(i, modelTokenSize);
        } else if (isTreeAttn()) {
            // Tree SpD candidate mask
            DCHECK_GE(modelTokenSize, mTreeMask.size())
                << "Tree mask height should be less than or equal to modelTokenSize";
            const size_t treeMaskLength = mTreeMask[0].size();
            // needHackOffset > 0 is designed for draft tree conduction process.
            // e.g. topk=4,2,1, tree shape =
            // [[root0], [1, 2, 3, 4], [5, 6], [7]] (From left to right, from top to bottom, and
            // the parent node of ith level is the top1 a.k.a first token of the i-1 th level)
            // Then the tree mask root0
            //        0
            //    -----
            //    0 | 1
            // the mask of [1, 2, 3, 4]
            //      | 0 | 1 2 3 4
            //    ----------------
            //    1 | 1 | 1 0 0 0
            //    2 | 1 | 0 1 0 0
            //    3 | 1 | 0 0 1 0
            //    4 | 1 | 0 0 0 1
            // the mask of [5, 6]
            //      | 0 |  1 2 3 4  | 5 6
            //    ---------------------
            //          <-needHack-->
            //    5 | 1 |  1 0 0 0  | 1 0
            //    6 | 1 |  1 0 0 0  | 0 1
            // we can observe that for the middle part is not the standard tree mask.

            // The mask of root 0 to inference
            DCHECK_GE(treeMaskLength, modelTokenSize)
                << "Tree mask width sould be greater than or equal to modelTokenSize";
            const size_t needHackOffset = treeMaskLength - modelTokenSize;
            // (needHackOffset = 0): The tree mask[:, :cacheSize] keeps unchanged
            // (needHackOffset > 0): The tree mask[:, :cacheSize] and mTreeMask
            //                       (cacheSize - needHackOffset to cacheSize + modelTokenSize)
            //                       overlaps on position cacheSize - needHackOffset to cacheSize
            //                       ==> need to hack the overlapping of two masks
            for (const auto maskVal : mTreeMask[rowIdx]) {
                // Hack padded attention and create new attention mask
                DCHECK_GE(i, needHackOffset);
                attnMaskRow[i++ - needHackOffset] = (maskVal == 1) ? maskTrue : maskFalse;
            }
            DCHECK_EQ(i - needHackOffset, modelTokenSize);
        }
    }

    // Modify mask for padding if needed. Mask is not updatable if modified for padding.
    mIsMaskUpdatable = !adjustMaskForPadding<MaskType>(modelTokenSize);
}

template <typename MaskType>
void MaskBuilder::updateMask(const size_t modelTokenSize, const size_t numSeenToken,
                             const size_t length) {
    if (!mIsMaskUpdatable) {
        buildMask<MaskType>(modelTokenSize, numSeenToken);
        return;
    }

    const size_t cacheMaskNumRows = mCacheMaskUseBroadcast ? 1u : modelTokenSize;
    const auto [cacheMaskBuffer, cacheMaskRowStride] = [&] {
        MaskType* cacheMaskBuffer = nullptr;
        size_t rowStride = 0; // Num vals per row
        if (mCacheMaskBuffer) {
            // Split-mask
            cacheMaskBuffer = reinterpret_cast<MaskType*>(mCacheMaskBuffer);
            rowStride = mCacheMaskSizeBytes / cacheMaskNumRows / kMaskTypeSize;
        } else {
            // Merged-mask
            cacheMaskBuffer = reinterpret_cast<MaskType*>(mMaskBuffer);
            rowStride = mMaskSizeBytes / modelTokenSize / kMaskTypeSize;
        }
        DCHECK(cacheMaskBuffer);
        DCHECK_GT(rowStride, 0);
        return std::pair{cacheMaskBuffer, rowStride};
    }();

    // Only modify the left rectangle part
    const size_t startValidCacheIdx = mCacheLength - std::min(mCacheLength, numSeenToken);
    const size_t trueCount = std::min(length, numSeenToken); // Can only True for seen token
    for (size_t rowIdx = 0; rowIdx < cacheMaskNumRows; rowIdx++) {
        auto cacheMaskRow = cacheMaskBuffer + rowIdx * cacheMaskRowStride + startValidCacheIdx;
        std::fill(cacheMaskRow, cacheMaskRow + trueCount, MaskVal<MaskType>::kTrue);
    }
    // Modify mask for padding if needed. Mask is not updatable if modified for padding.
    mIsMaskUpdatable = !adjustMaskForPadding<MaskType>(modelTokenSize);
}

void MaskBuilder::buildMask(const size_t modelTokenSize, const size_t numSeenToken) {
    DCHECK(mMaskBuffer || (mAttnMaskBuffer && mCacheMaskBuffer))
        << "Mask buffer is not yet set to build/update";

    switch (kMaskType) {
        case LLMType::INT8:
            buildMask<int8_t>(modelTokenSize, numSeenToken);
            return;
        case LLMType::INT16:
            buildMask<int16_t>(modelTokenSize, numSeenToken);
            return;
        case LLMType::FP16:
            buildMask<__fp16>(modelTokenSize, numSeenToken);
            return;
        case LLMType::FP32:
            buildMask<float>(modelTokenSize, numSeenToken);
            return;
        default:
            break;
    }
    LOG(FATAL) << "Attempting to build mask with type " << getLLMTypeName(kMaskType) << ". "
               << "Supported types are INT8, INT16, FP16, FP32.";
}

void MaskBuilder::updateMask(const size_t modelTokenSize, const size_t numSeenToken,
                             const size_t length) {
    DCHECK(mMaskBuffer || (mAttnMaskBuffer && mCacheMaskBuffer))
        << "Mask buffer is not yet set to build/update";

    switch (kMaskType) {
        case LLMType::INT8:
            updateMask<int8_t>(modelTokenSize, numSeenToken, length);
            return;
        case LLMType::INT16:
            updateMask<int16_t>(modelTokenSize, numSeenToken, length);
            return;
        case LLMType::FP16:
            updateMask<__fp16>(modelTokenSize, numSeenToken, length);
            return;
        case LLMType::FP32:
            updateMask<float>(modelTokenSize, numSeenToken, length);
            return;
        default:
            break;
    }
    LOG(FATAL) << "Attempting to update with an unsupported mask type. "
               << "Supported types are INT8, INT16, FP16, FP32.";
}

void MaskBuilder::notifyLeftPadding(const size_t padLength) {
    CHECK_EQ(mRightPadLength, 0) << "Attempting to set left pad after right pad has been set.";
    if (mLeftPadLength > 0) {
        LOG(WARN) << "Calling notifyLeftPadding() multiple times before building/updating mask.";
    }
    CHECK(padLength == 0 || !isFoldedGenBatchMode())
        << "Padding is not supported in folded gen batch mode.";
    mLeftPadLength = padLength;
}

void MaskBuilder::notifyRightPadding(const size_t padLength) {
    CHECK_EQ(mLeftPadLength, 0) << "Attempting to set right pad after left pad has been set.";
    if (mRightPadLength > 0) {
        LOG(WARN) << "Calling notifyRightPadding() multiple times before building/updating mask.";
    }
    CHECK(padLength == 0 || !isFoldedGenBatchMode())
        << "Padding is not supported in folded gen batch mode.";
    mRightPadLength = padLength;
}

template <typename MaskType>
bool MaskBuilder::adjustMaskForPadding(const size_t modelTokenSize) {
    if (mLeftPadLength + mRightPadLength == 0) {
        return false; // No need to modify mask since no padding
    }
    DCHECK(mLeftPadLength == 0 || mRightPadLength == 0)
        << "Only allow setting either left or right pad";

    constexpr auto maskFalse = MaskVal<MaskType>::kFalse;

    const auto [attnMaskBuffer, attnMaskRowStride] = [this, modelTokenSize] {
        MaskType* attnMaskBuffer = nullptr;
        size_t rowStride = 0; // Num vals per row
        if (mAttnMaskBuffer) {
            // Split-mask
            attnMaskBuffer = reinterpret_cast<MaskType*>(mAttnMaskBuffer);
            rowStride = mAttnMaskSizeBytes / modelTokenSize / kMaskTypeSize;
        } else {
            // Merged-mask
            attnMaskBuffer = reinterpret_cast<MaskType*>(mMaskBuffer) + mCacheLength;
            rowStride = mMaskSizeBytes / modelTokenSize / kMaskTypeSize;
        }
        return std::pair{attnMaskBuffer, rowStride};
    }();

    bool isMaskModified = false;

    if (mLeftPadLength > 0) {
        // Mask the padded attention columns
        for (size_t inTokIdx = 0; inTokIdx < modelTokenSize; inTokIdx++) {
            auto curAttnMaskBuf = attnMaskBuffer + inTokIdx * attnMaskRowStride;
            // Anything from inTokIdx + 1 onwards is already False, so can skip them.
            const size_t maskPadCount = std::min(mLeftPadLength, inTokIdx + 1);
            std::fill(curAttnMaskBuf, curAttnMaskBuf + maskPadCount, maskFalse);
        }
        mLeftPadLength = 0; // Reset pad length
        isMaskModified = true;
    } else if (mRightPadLength > 0) {
        // Mask the padded attention rows
        const auto startIdx = modelTokenSize - mRightPadLength;
        for (size_t inTokIdx = startIdx; inTokIdx < modelTokenSize; inTokIdx++) {
            auto curAttnMaskBuf = attnMaskBuffer + inTokIdx * attnMaskRowStride;
            // Set the entire row to false
            std::fill(curAttnMaskBuf, curAttnMaskBuf + modelTokenSize, maskFalse);
        }
        mRightPadLength = 0; // Reset pad length
        isMaskModified = true;
    }
    return isMaskModified;
}

void MaskBuilder::fillPadingForTreeMask(std::vector<std::vector<int>>& mask,
                                        const size_t modelTokenSize) {
    const size_t height = mask.size();
    if (height == modelTokenSize) {
        return;
    }
    const size_t width = mask[0].size();
    const size_t offset = modelTokenSize - height;
    const size_t widthAfterPadding = width + offset;
    const size_t heightAfterPadding = modelTokenSize;

    for (auto& row : mask) {
        row.resize(widthAfterPadding, 0);
    }
    mask.resize(heightAfterPadding, std::vector<int>(widthAfterPadding, 0));
}

void MaskBuilder::setTreeMask(const std::vector<std::vector<int>>& mask,
                              const size_t modelTokenSize) {
    mTreeMask = mask;
    fillPadingForTreeMask(mTreeMask, modelTokenSize);
}

void MaskBuilder::setMedusaTreeMask(const std::vector<std::vector<int>>& mask) {
    mMedusaTreeMask = mask;
}

void MaskBuilder::enterFoldedGenBatchMode(const size_t numPromptTokens) {
    DCHECK_GT(numPromptTokens, 0);
    mGenBatchNumPromptTokens = numPromptTokens;
}

void MaskBuilder::reset() {
    markMaskDirty();
    mMedusaTreeMask.clear();
    mGenBatchNumPromptTokens = 0;
    mTreeMask.clear();
}

} // namespace mtk::llm_helper