#pragma once
#include "common/logging.h"
#include "executor/llm_executor.h"
#include "executor/llm_medusa_executor.h"
#include "executor/llm_ringbuffer_executor.h"
#include "executor/neuron_executor.h"
#include "executor/neuron_usdk_executor.h"
#include "executor/tflite_executor.h"

#include <type_traits>

namespace mtk {

#ifdef USE_USDK_BACKEND
using NeuronExecutorType = NeuronUsdkExecutor;
#else
using NeuronExecutorType = NeuronExecutor;
#endif

#ifdef DISABLE_RING_BUFFER
using LlmExecutorType = LlmExecutor;
#else
using LlmExecutorType = LlmRingBufferExecutor;
#endif

using LlmMedusaExecutorType = LlmMedusaExecutor;

using TFLiteExecutorType = TfliteExecutor;

#define GetExecutorClass(ExecKind) mtk::ExecKind##ExecutorType

enum class ExecutorKind {
    Neuron,
    TFLite,
    Llm,
    LlmMedusa
};

class ExecutorFactory {
public:
    explicit ExecutorFactory(const ExecutorKind executorKind = ExecutorKind::Llm)
        : mExecutorKind(executorKind) {}

    ExecutorFactory& setType(const ExecutorKind executorKind) {
        mExecutorKind = executorKind;
        return *this;
    }

    // The created executor object will be cast to `std::unique_ptr<ReturnType>`, which defaults to
    // `std::unique_ptr<Executor>`.
    template <typename ReturnType = Executor, typename... Args>
    std::unique_ptr<ReturnType> create(Args&&... args) const {
#define __DECL__(ExecKind)                                                                        \
    case ExecutorKind::ExecKind: {                                                                \
        using ExecutorType = ExecKind##ExecutorType;                                              \
        if constexpr (std::is_base_of_v<ReturnType, ExecutorType>) {                              \
            auto executor = createImpl<ExecutorType>(std::forward<Args>(args)...);                \
            DCHECK(executor != nullptr)                                                           \
                << "Unable to create '" #ExecKind "' executor with the given " << sizeof...(Args) \
                << " arguments.";                                                                 \
            return executor;                                                                      \
        } else {                                                                                  \
            return nullptr;                                                                       \
        }                                                                                         \
    }

        switch (mExecutorKind) {
            __DECL__(Neuron)
            __DECL__(TFLite)
            __DECL__(Llm)
            __DECL__(LlmMedusa)
        }

#undef __DECL__
    }

private:
    // Can be constructed with the provided arguments
    template <typename ExecutorType, typename... Args>
    static std::enable_if_t<std::is_constructible_v<ExecutorType, Args...>,
                            std::unique_ptr<ExecutorType>>
    createImpl(Args&&... args) {
        return std::make_unique<ExecutorType>(std::forward<Args>(args)...);
    }

    // Cannot be constructed with the provided arguments
    template <typename ExecutorType, typename... Args>
    static std::enable_if_t<!std::is_constructible_v<ExecutorType, Args...>,
                            std::unique_ptr<ExecutorType>>
    createImpl(Args&&... args) {
        return nullptr;
    }

private:
    ExecutorKind mExecutorKind;
};

} // namespace mtk