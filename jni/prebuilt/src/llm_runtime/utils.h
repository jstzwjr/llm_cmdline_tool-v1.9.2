#pragma once

#include "llm_helper/include/utils.h"

#include "mtk_llm_options.h"

#include <regex>
#include <string>
#include <unordered_set>

namespace mtk::llm_runtime::utils {

inline size_t getNumChunks(const LlmRuntimeOptions& runtimeOptions) {
    std::unordered_set<size_t> numChunkSet;
    for (const auto& [_, files] : runtimeOptions.dlaFiles) {
        numChunkSet.insert(files.size());
    }
    CHECK_LE(numChunkSet.size(), 1) << "Inconsistent number of dla chunks found in runtimeOptions.";
    return numChunkSet.empty() ? 0 : *numChunkSet.cbegin();
}

inline std::tuple<size_t, size_t, size_t> parseModelConfig(const std::string& modelConfig,
                                                           const LlmModelOptions& modelOptions) {
    const std::regex tokenSizePat("([0-9]+)[tT]");
    const std::regex cacheSizePat("([0-9]+)[cC]");
    size_t tokenSize = 0, cacheSize = 0;
    std::smatch match;

    // Parse token size, and must be provided.
    if (std::regex_search(modelConfig, match, tokenSizePat))
        tokenSize = std::stoi(match[0].str());
    else
        LOG(FATAL) << "Token size is not provided in 'dlaPaths' model config.";

    // Parse cache size, will take from modelOptions if not provided.
    if (std::regex_search(modelConfig, match, cacheSizePat))
        cacheSize = std::stoi(match[0].str());
    else
        cacheSize = modelOptions.cacheSize;

    // Only support literal batch size for 1t gen model.
    // For folded batch gen mode, batchSize value is set to 1, and tokenSize is interpreted as
    // "effective batch size" during inference.
    const size_t batchSize = (tokenSize == 1) ? modelOptions.genModelBatchSize : 1;

    return {batchSize, tokenSize, cacheSize};
}

template <typename T>
inline bool allSameSize(const T& iterable) {
    return mtk::llm_helper::allSame(iterable, [](const auto& item) { return item.size(); });
}

// Clipped subtraction between unsigned values
template <typename T, typename U>
inline T max0Subtract(const T lhs, const U rhs) {
    if (lhs < rhs)
        return 0UL;
    return lhs - rhs;
}

using TokenType = Tokenizer::TokenType;

// Partition input tokens into subtoken intervals by a set of delimiter tokens
inline std::vector<std::pair<size_t, size_t>>
partitionTokens(const std::vector<TokenType>& inputTokens, const std::vector<TokenType>& delimiters,
                const bool preserveDelimiter = true) {
    std::vector<std::pair<size_t, size_t>> result; // Intervals

    auto appendResult = [&result](const auto start, const auto end) {
        if (start != end)
            result.push_back({start, end});
    };

    auto find_not = [](const auto begin, const auto end, const auto& ref) {
        auto it = begin;
        while (it != end && *it == ref) {
            ++it;
        }
        return it;
    };

    // A list of pair of delimiter index and repeat count
    std::vector<std::pair<size_t, size_t>> delimIndexes;

    for (const auto delim : delimiters) {
        const auto begin = inputTokens.begin();
        const auto end = inputTokens.end();
        auto it = begin;

        while (it != end) {
            auto delimStart = std::find(it, end, delim);

            if (delimStart == end) {
                break;
            }

            // Find the next non-delim token to skip any repeats
            auto delimEnd = find_not(std::next(delimStart), end, delim);

            const auto delimIdx = delimStart - begin;
            const auto numRepeats = delimEnd - delimStart;
            delimIndexes.emplace_back(delimIdx, numRepeats);

            it = delimEnd;
        }
    }
    std::sort(delimIndexes.begin(), delimIndexes.end(),
              [](const auto& left, const auto& right) { return left.first < right.first; });

    size_t startIdx = 0;
    for (const auto& [delimIdx, numDelimRepeats] : delimIndexes) {
        DCHECK_LE(delimIdx, inputTokens.size());
        DCHECK_GE(numDelimRepeats, 1);
        const auto nextStartIdx = delimIdx + numDelimRepeats;

        appendResult(startIdx, delimIdx);
        if (preserveDelimiter) {
            appendResult(delimIdx, nextStartIdx);
        }
        startIdx = nextStartIdx;
    }
    appendResult(startIdx, inputTokens.size()); // Until the end
    return result;
}

} // namespace mtk::llm_runtime::utils