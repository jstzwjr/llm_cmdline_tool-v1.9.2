#include "executor/llm_medusa_executor.h"

#include "common/logging.h"
#include "common/scope_profiling.h"

namespace mtk {

void LlmMedusaExecutor::setPosEmbed() {
    const auto tokenIndex = getTokenIndex();
    // Cut the array from master
    if (tokenIndex >= kMaxTokenLength) {
        LOG(FATAL) << "Attempting to set rotaty embedding using index exceeding the supported "
                      "max token length ("
                   << kMaxTokenLength << ")";
    }
    DLOG_FUNC_LATENCY(ms)
    DCHECK_EQ(getNumInputsFor(IOKind::RotEmb), kRotEmbInputCount);

    auto getRotEmbInputs = [&]() {
        std::vector<void*> rotEmbInputs(kRotEmbInputCount);
        for (size_t i = 0; i < kRotEmbInputCount; i++)
            rotEmbInputs[i] = this->getInputBuffer(IOKind::RotEmb, i);
        return rotEmbInputs;
    };

    const bool isMedusaTreeAttn = !mMedusaTreePositions.empty();

    if (isMedusaTreeAttn) {
        CHECK_EQ(mMedusaTreePositions.size(), mModelTokenSize)
            << "Medusa tree attention is not set.";
        DCHECK_EQ(getLeftPadding(), 0);
        DCHECK_EQ(getRightPadding(), 0);
        mRotEmbMasterLut->setEmbed(getRotEmbInputs(), tokenIndex, mMedusaTreePositions);
    } else {
        mRotEmbMasterLut->setEmbed(
            getRotEmbInputs(), tokenIndex, mModelTokenSize, getLeftPadding(), getRightPadding());
    }
}

void LlmMedusaExecutor::resetTokenIndex(const size_t tokenIndex) {
    LlmExecutorBase::resetTokenIndex(tokenIndex);
    resetMedusaTreeAttn();
}

void LlmMedusaExecutor::setMedusaTreeAttn(const std::vector<std::vector<int>>& mask,
                                          const std::vector<size_t>& positions) {
    mMedusaTreePositions = positions;
    mMaskBuilder->setMedusaTreeMask(mask);
}

void LlmMedusaExecutor::resetMedusaTreeAttn() {
    mMedusaTreePositions.clear();
}

} // namespace mtk