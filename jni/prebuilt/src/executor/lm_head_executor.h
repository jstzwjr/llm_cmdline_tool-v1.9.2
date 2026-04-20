#pragma once

#include "common/file_source.h"
#include "executor/neuron_executor.h"
#include "executor/neuron_usdk_executor.h"
#include "executor/shared_weights.h"

namespace mtk {

#ifdef USE_USDK_BACKEND
using ExecutorBackend = NeuronUsdkExecutor;
#else
using ExecutorBackend = NeuronExecutor;
#endif

class LmHeadExecutor : public ExecutorBackend {
public:
    explicit LmHeadExecutor(const FileSource& lmHeadModelFile,
                            const SharedWeights& lmHeadSharedWeights = {});

    void initialize() override;
};

} // namespace mtk