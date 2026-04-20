#pragma once

#include "common/logging.h"
#include "mtk_llm.h"
#include "mtk_llm_types.h"
#include "tokenizer/tokenizer.h"

#include <iostream>
#include <queue>
#include <sstream>
#include <string>

#define ENSURE_NEXT_ARG_EXISTS(curArgIdx)                                              \
    if (curArgIdx + 1 >= argc) {                                                       \
        std::cout << "No value provided for for argument '" << argv[curArgIdx] << "'." \
                  << std::endl;                                                        \
        continue;                                                                      \
    }

namespace utils {

using TokenType = mtk::Tokenizer::TokenType;
using ArgmaxProb = std::pair<TokenType, float>;
using ArgmaxProbs = std::pair<TokenType, std::vector<float>>;

bool matchArgument(const std::string& target, const std::string& argPattern,
                   const std::string& argPatternShort = "", bool normalizeUnderscore = true);

bool isWhiteLine(const std::string& line);

class UTF8CharResolver {
public:
    UTF8CharResolver() {}
    bool addBytes(const std::string& byteStr);
    bool hasResolved();
    std::string getResolvedStr();
    static size_t utf8Len(char src);
    static size_t getUTF8FullLength(const std::string& str, size_t start_index = 0);

private:
    void setResolved() {
        mResolved = mAccum;
        mAccum.clear();
    }
    void setResolvedPartial(size_t unresolvedSize) {
        size_t resolved_size = mAccum.size() - unresolvedSize;
        mResolved = mAccum.substr(0, resolved_size);
        mAccum = mAccum.substr(resolved_size);
    }

private:
    size_t mUtfLengthRemaining = 0;
    bool mConcatMultibyteMode = false;
    std::string mAccum;
    std::string mResolved;
};

// Logits processors
float* repeatPenalty(float* logits, TokenType input_token, const float repetition_penalty = 1.2);

int16_t* suppressLogits(int16_t* logits, const std::vector<TokenType>& tokenIds);
float* suppressLogits(float* logits, const std::vector<TokenType>& tokenIds);

size_t samplingFromDistribution(const std::vector<float>& probs);

size_t samplingFromDistribution(const float* probs, const size_t size);

template <typename LogitsType>
std::pair<TokenType, LogitsType> argmaxWithMax(const LogitsType* array, const size_t vocabSize) {
    // Find argmax and corresponding value
    LogitsType max = array[0];
    size_t index = 0;
    for (size_t i = 1; i < vocabSize; i++) {
        const auto current = array[i];
        if (current > max) {
            index = i;
            max = current;
        }
    }
    return {index, max};
}

void convertToSoftmax(float* array, const size_t vocabSize, const float max,
                      const float temperature);

void convertToSoftmax(float* array, const size_t vocabSize, const float temperature = 0.0f);

template <typename LogitsType>
void makeSoftmax(std::vector<float>& softmaxBuffer, const LogitsType* logits,
                 const size_t vocabSize, const float temperature, const float qscale = 1.0f);

void makeSoftmax(std::vector<float>& softmaxBuffer, const mtk::LLMType logitsType,
                 const void* logits, const size_t vocabSize, const float temperature,
                 const float qscale = 1.0f);

TokenType argmaxFrom16bitLogits(const mtk::LLMType logitsType, const void* logits,
                                const size_t vocabSize);
TokenType argmaxFrom16bitLogits(const int16_t* logits, const size_t vocabSize);
TokenType argmaxFrom16bitLogits(const __fp16* logits, const size_t vocabSize);
ArgmaxProb argmaxProbFrom16bitLogits(const mtk::LLMType logitsType, const void* logits,
                                     const size_t vocabSize, const float modelOutputQuantScale);
ArgmaxProb argmaxProbFrom16bitLogits(const int16_t* logits, const size_t vocabSize,
                                     const float modelOutputQuantScale);
ArgmaxProb argmaxProbFrom16bitLogits(const __fp16* logits, const size_t vocabSize);
ArgmaxProb randomSampleFrom16bitLogits(const mtk::LLMType logitsType, const void* logits,
                                       const size_t vocabSize, const float modelOutputQuantScale,
                                       const float temperature);
ArgmaxProb randomSampleFrom16bitLogits(const int16_t* logits, const size_t vocabSize,
                                       const float modelOutputQuantScale, const float temperature);
ArgmaxProb randomSampleFrom16bitLogits(const __fp16* logits, const size_t vocabSize,
                                       const float tempertature);
ArgmaxProb argmaxProbFrom16bitLogits(const mtk::LLMType logitsType, const void* logits,
                                     const size_t vocabSize, const float modelOutputQuantScale,
                                     const TokenType tokenIdForProb);
ArgmaxProb argmaxProbFrom16bitLogits(const int16_t* logits, const size_t vocabSize,
                                     const float modelOutputQuantScale,
                                     const TokenType tokenIdForProb);
ArgmaxProb argmaxProbFrom16bitLogits(const __fp16* logits, const size_t vocabSize,
                                     const TokenType tokenIdForProb);
ArgmaxProb randomSampleFrom16bitLogits(const mtk::LLMType logitsType, const void* logits,
                                       const size_t vocabSize, const float modelOutputQuantScale,
                                       const float temperature, const TokenType tokenIdForProb);
ArgmaxProb randomSampleFrom16bitLogits(const int16_t* logits, const size_t vocabSize,
                                       const float modelOutputQuantScale, const float temperature,
                                       const TokenType tokenIdForProb);
ArgmaxProb randomSampleFrom16bitLogits(const __fp16* logits, const size_t vocabSize,
                                       const float temperature, const TokenType tokenIdForProb);
ArgmaxProbs randomSampleFrom16bitLogits(const mtk::LLMType logitsType, const void* logits,
                                        const size_t vocabSize, const float modelOutputQuantScale,
                                        const float temperature,
                                        const std::vector<TokenType>& tokensIdForProb);
ArgmaxProbs randomSampleFrom16bitLogits(const int16_t* logits, const size_t vocabSize,
                                        const float modelOutputQuantScale, const float temperature,
                                        const std::vector<TokenType>& tokensIdForProb);
ArgmaxProbs randomSampleFrom16bitLogits(const __fp16* logits, const size_t vocabSize,
                                        const float temperature,
                                        const std::vector<TokenType>& tokensIdForProb);
TokenType argmaxFromAdjustDistSpecDec(const mtk::LLMType LogitsType, const void* targetLogits,
                                      const void* draftLogits, const size_t vocabSize,
                                      const float targetOutputQuantScale,
                                      const float draftOutputQuantScale);
template <typename LogitsType>
TokenType argmaxFromAdjustDistSpecDec(const LogitsType* targetLogits, const LogitsType* draftLogits,
                                      const size_t vocabSize, const float targetOutputQuantScale,
                                      const float draftOutputQuantScale);
TokenType randomSampleFromAdjustDistSpecDec(const mtk::LLMType LogitsType, const void* targetLogits,
                                            const void* draftLogits, const size_t vocabSize,
                                            const float targetOutputQuantScale,
                                            const float draftOutputQuantScale,
                                            const float targetSamplingTemperature,
                                            const float draftSamplingTemperature);
template <typename LogitsType>
TokenType randomSampleFromAdjustDistSpecDec(const LogitsType* targetLogits,
                                            const LogitsType* draftLogits, const size_t vocabSize,
                                            const float targetOutputQuantScale,
                                            const float draftOutputQuantScale,
                                            const float targetSamplingTemperature,
                                            const float draftSamplingTemperature);

std::vector<TokenType> getTopkArgmaxV2(const mtk::LLMType logitsType, const void* logits,
                                       const size_t vocabSize, const size_t k);

template <typename LogitsType>
std::vector<size_t> getTopkArgmax(const LogitsType* logits, const size_t vocabSize, const size_t k);

std::pair<std::vector<TokenType>, std::vector<float>>
getTopkArgmaxAndLogitsV2(const mtk::LLMType logitsType, const void* logits, const size_t vocabSize,
                         const size_t k, const float modelOutputQuantScale,
                         const float temperature = 1.0f);

std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>>
conductTreeMask(const std::vector<int>& parentId);

template <typename ValueType, typename IndicesType>
std::vector<ValueType> gather(const std::vector<ValueType>& values,
                              const std::vector<IndicesType>& indices) {
    std::vector<ValueType> result;
    result.reserve(indices.size());

    const auto numel = values.size();

    for (auto index : indices) {
        DCHECK_LT(index, numel) << "Indexing out of range.";
        result.push_back(values[index]);
    }
    return result;
}

template <typename ValueType, typename IndicesType>
std::vector<ValueType> gather(const ValueType* values, const std::vector<IndicesType>& indices,
                              const size_t numel) {
    std::vector<float> result;
    result.reserve(indices.size());

    for (auto index : indices) {
        DCHECK_LT(index, numel) << "Indexing out of range.";
        result.push_back(values[index]);
    }
    return result;
}

std::vector<std::vector<int>> horizontalConcat(const std::vector<std::vector<int>>& leftMask,
                                               const std::vector<std::vector<int>>& rightMask);

// Preformatters
bool addPreformatter(const std::string& prefName, std::string& prompt);
std::string addPreformatter_AlpacaNoInput(const std::string& prompt);
std::string addPreformatter_OneShotConversation(const std::string& prompt);
std::string addPreformatter_VicunaNoInput(const std::string& prompt);
std::string addPreformatter_QwenNoInput(const std::string& prompt);
std::string addPreformatter_Qwen3NoInput(const std::string& prompt);
std::string addPreformatter_Qwen3NoInputNoThink(const std::string& prompt);
std::string addPreformatter_Llama3NoInput(const std::string& prompt);
std::string addPreformatter_Phi3NoInput(const std::string& prompt);
std::string addPreformatter_MinicpmNoInput(const std::string& prompt);
std::string addPreformatter_MinicpmNoInputZh(const std::string& prompt);
std::string addPreformatter_InternLM2(const std::string& prompt);
std::string addPreformatter_GemmaNoInput(const std::string& prompt);

std::vector<std::string> split(const std::string& str, const std::string& sep);

std::vector<TokenType> parseTokenString(const std::string& tokenString);

std::vector<std::string> readPromptFiles(const std::vector<std::string>& promptPaths,
                                         const bool onePromptPerLine);

} // namespace utils