#pragma once

#include "executor/llm_executor.h"
#include "executor/llm_ringbuffer_executor.h"

#include <vector>

namespace mtk {

#ifdef DISABLE_RING_BUFFER
using LlmExecutorBase = LlmExecutor;
#else
using LlmExecutorBase = LlmRingBufferExecutor;
#endif

class LlmMedusaExecutor : public LlmExecutorBase {
public:
    // Inherit parent class constructor
    using LlmExecutorBase::LlmExecutorBase;

    // Override functions
    virtual void setPosEmbed() override;

    virtual void resetTokenIndex(const size_t tokenIndex) override;

    // Medusa
    void setMedusaTreeAttn(const std::vector<std::vector<int>>& mask,
                           const std::vector<size_t>& positions);

    void resetMedusaTreeAttn();

private:
    std::vector<size_t> mMedusaTreePositions;
};

} // namespace mtk