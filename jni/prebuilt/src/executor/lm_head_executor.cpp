#include "executor/lm_head_executor.h"

#include "common/thread_pool.h"

namespace mtk {

LmHeadExecutor::LmHeadExecutor(const FileSource& lmHeadModelFile,
                               const SharedWeights& lmHeadSharedWeights)
    : ExecutorBackend(lmHeadModelFile, lmHeadSharedWeights) {}

void LmHeadExecutor::initialize() {
    // Define IOs
    setNumInputs(isSharedWeightsUsed() ? 2 : 1);
    setNumOutputs(1);

    // Load LM Head model
    BasicThreadPool threadPool(2);
    if (isSharedWeightsUsed()) {
        const size_t sharedWeightsInputIdx = 1;
        reserveInputBuffer(sharedWeightsInputIdx);
        initAllocator(); // Allocator must be ready before loadSharedWeights is invoked
        threadPool.push([this] { loadSharedWeights(sharedWeightsInputIdx); });
    }
    threadPool.push([this] { ExecutorBackend::initialize(); });
    threadPool.joinAll();
}

} // namespace mtk