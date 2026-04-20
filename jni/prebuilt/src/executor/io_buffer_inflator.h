#pragma once

#include "allocator.h"

#include <algorithm>
#include <functional>
#include <vector>

// clang-format off

namespace mtk {

template <class Scenario>
class IoBufferInflator {
public:
    IoBufferInflator(const std::vector<Scenario>& scenarios, const Scenario curScenario)
        : kScenarios(scenarios),
          kCurScenario(curScenario),
          kNormalizerBase(curScenario.batchSize * curScenario.tokenSize * curScenario.cacheSize) {}

    // Find the scenario that maximizes the size
    void findMaxSizeScenario() {
        auto key = [&](const auto& left, const auto& right) {
            auto getRelevantSize = [&](const auto& scenario) {
                size_t prod = 1;
                if (mUseBatchSize) prod *= scenario.batchSize;
                if (mUseTokenSize) prod *= scenario.tokenSize;
                if (mUseCacheSize) prod *= scenario.cacheSize;
                for (const auto& calcFunc : mCustomSizeCalcFuncs) {
                    prod *= calcFunc(scenario);
                }
                return prod;
            };
            return getRelevantSize(left) < getRelevantSize(right);
        };
        const auto& maxSizeScenario = *std::max_element(kScenarios.begin(), kScenarios.end(), key);

        // Select max size scenario size if depends on it,
        // otherwise select current scenario size to cancel out the term in kNormalizerBase
        mMultiplier = 1;
        mMultiplier *= mUseBatchSize ? maxSizeScenario.batchSize : kCurScenario.batchSize;
        mMultiplier *= mUseTokenSize ? maxSizeScenario.tokenSize : kCurScenario.tokenSize;
        mMultiplier *= mUseCacheSize ? maxSizeScenario.cacheSize : kCurScenario.cacheSize;
        // Default normalizer
        mNormalizer = kNormalizerBase;

        // Custom multiplier terms for max size scenario
        for (const auto& calcFunc : mCustomSizeCalcFuncs) {
            mMultiplier *= calcFunc(maxSizeScenario);
        }
        // Custom normalizer terms for current scenario
        for (const auto& calcFunc : mCustomSizeCalcFuncs) {
            mNormalizer *= calcFunc(kCurScenario);
        }
    }

    void inflate(IOBuffer& ioBuf) {
        if (mMultiplier == mNormalizer) {
            return;
        }
        const auto oldSize = ioBuf.sizeBytes;
        const auto newSize = ioBuf.usedSizeBytes * mMultiplier / mNormalizer;
        ioBuf.sizeBytes = newSize;
        LOG(DEBUG) << "Reassigned required allocation size: " << oldSize << " -> " << newSize;
    }

    IoBufferInflator& useBatchSize() {
        mUseBatchSize = true;
        return *this;
    }

    IoBufferInflator& useTokenSize() {
        mUseTokenSize = true;
        return *this;
    }

    IoBufferInflator& useCacheSize() {
        mUseCacheSize = true;
        return *this;
    }

    IoBufferInflator& useCustomSize(std::function<size_t(const Scenario&)> sizeCalcFn) {
        mCustomSizeCalcFuncs.push_back(std::move(sizeCalcFn));
        return *this;
    }

    void resetUses() {
        mUseBatchSize = false;
        mUseTokenSize = false;
        mUseCacheSize = false;
        mCustomSizeCalcFuncs.clear();
    }

private:
    const std::vector<Scenario>& kScenarios;
    const Scenario kCurScenario;

    const size_t kNormalizerBase;

    size_t mNormalizer = 1;
    size_t mMultiplier = 1;

    bool mUseBatchSize = false;
    bool mUseTokenSize = false;
    bool mUseCacheSize = false;
    std::vector<std::function<size_t(const Scenario&)>> mCustomSizeCalcFuncs;
};

} // namespace mtk