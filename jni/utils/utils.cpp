#include "utils.h"

#include "common/file_source.h"
#include "common/logging.h"

#include <cctype>
#include <fstream>
#include <queue>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace utils {

using mtk::LLMType;

static const size_t randomSeed = 12345678;

// Local helper template functions
namespace {
// clang-format off
template <typename T>
struct is_float_or_fp16
    : std::integral_constant<bool, std::is_floating_point_v<T> ||
                                   std::is_same_v<std::remove_cv_t<T>, __fp16>> {};
// clang-format on

template <typename T>
inline constexpr bool is_float_or_fp16_v = is_float_or_fp16<T>::value;

template <typename T, typename U = void>
struct To32BitTypeImpl {
    static_assert(std::is_arithmetic_v<T>);
    using type = int32_t;
};

template <typename T>
struct To32BitTypeImpl<T, typename std::enable_if_t<is_float_or_fp16_v<T>>> {
    using type = float;
};

template <typename T>
using To32BitType = typename To32BitTypeImpl<T>::type;

static std::vector<std::vector<char>> bufferPool(1);

template <typename OutType, size_t bufIdx = 0, typename InType = void>
inline auto makeCopy(const InType* array, const size_t numel, const float scale = 1.0f) {
    constexpr auto targetTypeSize = sizeof(OutType);
    if (bufIdx >= bufferPool.size()) {
        bufferPool.resize(bufIdx + 1);
    }
    auto& buffer = bufferPool[bufIdx];
    if (buffer.size() / targetTypeSize < numel) {
        buffer.resize(numel * targetTypeSize);
    }
    auto tmp = reinterpret_cast<OutType*>(buffer.data());
    if (is_float_or_fp16_v<OutType> && scale != 1.0) {
        for (size_t i = 0; i < numel; i++) {
            tmp[i] = array[i] * scale;
        }
    } else {
        for (size_t i = 0; i < numel; i++) {
            tmp[i] = array[i];
        }
    }
    return tmp;
}

template <size_t bufIdx = 0, typename T = void>
inline auto make32BitCopy(const T* array, const size_t numel, const float scale = 1.0f) {
    return makeCopy<To32BitType<T>, bufIdx>(array, numel, scale);
}

template <size_t bufIdx = 0, typename T = void>
inline float* makeFP32Copy(const T* array, const size_t numel, const float scale = 1.0f) {
    return makeCopy<float, bufIdx>(array, numel, scale);
}

template <size_t bufIdx = 0>
inline void* make32BitCopy(const LLMType type, const void* array, const size_t numel,
                           const float scale = 1.0f) {
    switch (type) {
        case LLMType::INT8:
            return make32BitCopy<bufIdx>(reinterpret_cast<const int8_t*>(array), numel, scale);
        case LLMType::INT16:
            return make32BitCopy<bufIdx>(reinterpret_cast<const int16_t*>(array), numel, scale);
        case LLMType::INT32:
            return make32BitCopy<bufIdx>(reinterpret_cast<const int*>(array), numel, scale);
        case LLMType::FP16:
            return make32BitCopy<bufIdx>(reinterpret_cast<const __fp16*>(array), numel, scale);
        case LLMType::FP32:
            return make32BitCopy<bufIdx>(reinterpret_cast<const float*>(array), numel, scale);
        default:
            LOG(FATAL) << "Unsupported type: " << mtk::getLLMTypeName(type);
    }
    return nullptr;
}

template <size_t bufIdx = 0, typename LogitsType = void>
float* makeTempSoftmax(const LogitsType* logits, const size_t vocabSize, const float temperature,
                       const float qscale = 1.0f) {
    auto tmp = makeFP32Copy<bufIdx>(logits, vocabSize, qscale);
    convertToSoftmax(tmp, vocabSize, temperature);
    return tmp;
}

template <size_t bufIdx = 0>
float* makeTempSoftmax(const LLMType logitsType, const void* logits, const size_t vocabSize,
                       const float temperature, const float qscale) {
    switch (logitsType) {
        case LLMType::INT16:
            return makeTempSoftmax<bufIdx>(
                reinterpret_cast<const int16_t*>(logits), vocabSize, temperature, qscale);
        case LLMType::FP16:
            return makeTempSoftmax<bufIdx>(
                reinterpret_cast<const __fp16*>(logits), vocabSize, temperature);

        case LLMType::FP32:
            return makeTempSoftmax<bufIdx>(
                reinterpret_cast<const float*>(logits), vocabSize, temperature);

        default:
            LOG(ERROR) << "makeTempSoftmax only supports INT16/FP16/FP32 logits.";
            return nullptr;
    }
}

inline bool isFloating(const LLMType type) {
    switch (type) {
        case LLMType::FP16:
        case LLMType::FP32:
            return true;
        default:
            return false;
    }
}

} // namespace

bool matchArgument(const std::string& target, const std::string& argPattern,
                   const std::string& argPatternShort, bool normalizeUnderscore) {
    // Replace '_' with '-'
    auto normalize = [&](const std::string& arg) {
        return normalizeUnderscore ? std::regex_replace(arg, std::regex("_"), "-") : arg;
    };
    auto getRegexPattern = [&]() {
        const auto argPatternNorm = normalize(argPattern);
        const auto argPatternShortNorm = normalize(argPatternShort);
        if (argPatternShort.size() > 0) {
            std::ostringstream patStream;
            patStream << "(" << argPatternNorm << ")|(" << argPatternShortNorm << ")";
            return std::regex(patStream.str());
        } else {
            return std::regex(argPatternNorm);
        }
    };
    return std::regex_match(normalize(target), getRegexPattern());
}

bool isWhiteLine(const std::string& line) {
    return (line.size() == 1 && (line[0] == '\n' || line[0] == '\r'));
}

bool UTF8CharResolver::hasResolved() {
    return mResolved.size() > 0;
}

std::string UTF8CharResolver::getResolvedStr() {
    if (mConcatMultibyteMode)
        return "";
    return mResolved;
}

size_t UTF8CharResolver::utf8Len(char src) {
    const size_t lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

size_t UTF8CharResolver::getUTF8FullLength(const std::string& str, size_t startIndex) {
    size_t strLength = str.size();
    size_t curIdx = startIndex;
    size_t totalLength = 0;
    while (curIdx < strLength) {
        size_t curUtf8Length = utf8Len(str[curIdx]);
        curIdx += curUtf8Length;
        totalLength += curUtf8Length;
    }
    return totalLength;
}

bool UTF8CharResolver::addBytes(const std::string& byteStr) {
    size_t curStrLength = byteStr.size();
    LOG(DEBUG) << "UTF8: mConcatMultibyteMode=" << mConcatMultibyteMode
               << ", mUtfLengthRemaining=" << mUtfLengthRemaining
               << ", mAccum.size()=" << mAccum.size() << ", curStrLength=" << curStrLength;
    if (!mConcatMultibyteMode) {
        mAccum = byteStr;
        LOG(DEBUG) << "UTF8: Concat mode false, check multibyte";
        mUtfLengthRemaining = getUTF8FullLength(byteStr);

        if (mUtfLengthRemaining > curStrLength) {
            // Start concat mode
            LOG(DEBUG) << "UTF8: Set concat mode true";
            mConcatMultibyteMode = true;
            mUtfLengthRemaining -= curStrLength;
        } else if (mUtfLengthRemaining < curStrLength) {
            LOG(FATAL) << "UTF8: Unreachable case: mUtfLengthRemaining < curStrLength ("
                       << mUtfLengthRemaining << " < " << curStrLength << ")";
        } else {
            if (curStrLength == 0) { // bos or eos, just update
                LOG(DEBUG) << "UTF8: bos/eos, just update";
            } else if (mUtfLengthRemaining == curStrLength) { // Is sub-byte
                LOG(DEBUG) << "UTF8: token == byte, update";
            }
            mUtfLengthRemaining = 0;
            setResolved();
            return true;
        }
    } else {
        mAccum += byteStr;
        LOG(DEBUG) << "UTF8: Concat mode true. Concat response: " << mAccum;

        if (mUtfLengthRemaining == curStrLength) {
            // Exit concat
            LOG(DEBUG) << "UTF8: Concat done, update. Set concat mode to false";
            mUtfLengthRemaining = 0;
            mConcatMultibyteMode = false;
            setResolved();
            return true;
        } else if (mUtfLengthRemaining < curStrLength) {
            // The current byteStr overshoots what was required to form a usable utf8 word
            // Extract the resolved part from `mAccum` to `resolved`
            setResolvedPartial(curStrLength - mUtfLengthRemaining);
            mUtfLengthRemaining = getUTF8FullLength(byteStr, mUtfLengthRemaining);
            return true; // next round will continue in concat mode
        } else {
            // continue to concat
            mUtfLengthRemaining -= curStrLength;
            LOG(DEBUG) << "UTF8: utf_len=" << mUtfLengthRemaining
                       << ", curStrLength=" << curStrLength;
        }
    }
    mResolved.clear();
    return false;
}

float* repeatPenalty(float* logits, TokenType tokenId, const float repetition_penalty) {
    if (tokenId < 0)
        return logits;

    float score = logits[tokenId];
    if (score < 0)
        score *= repetition_penalty;
    else
        score /= repetition_penalty;
    logits[tokenId] = score;

    return logits;
}

int16_t* suppressLogits(int16_t* logits, const std::vector<TokenType>& tokenIds) {
    for (const auto tokenId : tokenIds) {
        if (tokenId < 0)
            continue;
        logits[tokenId] = INT16_MIN;
    }
    return logits;
}

float* suppressLogits(float* logits, const std::vector<TokenType>& tokenIds) {
    for (const auto tokenId : tokenIds) {
        if (tokenId < 0)
            continue;
        logits[tokenId] = -FLT_MAX;
    }
    return logits;
}

size_t samplingFromDistribution(const std::vector<float>& probs) {
    std::random_device device;
    std::mt19937 engine(device());
    engine.seed(randomSeed);
    std::discrete_distribution<> dist(probs.begin(), probs.end());
    const size_t sampledIndex = dist(engine);
    return sampledIndex;
}

size_t samplingFromDistribution(const float* probs, const size_t size) {
    std::random_device device;
    std::mt19937 engine(device());
    engine.seed(randomSeed);
    std::discrete_distribution<> dist(probs, probs + size);
    const size_t sampledIndex = dist(engine);
    return sampledIndex;
}

TokenType argmaxFrom16bitLogits(const LLMType logitsType, const void* logits,
                                const size_t vocabSize) {
    switch (logitsType) {
        case LLMType::INT16:
            return argmaxFrom16bitLogits(reinterpret_cast<const int16_t*>(logits), vocabSize);
        case LLMType::FP16:
            return argmaxFrom16bitLogits(reinterpret_cast<const __fp16*>(logits), vocabSize);
        default:
            LOG(ERROR) << "argmaxFrom16bitLogits function only supports INT16 and FP16 logits.";
            return 0;
    }
}

TokenType argmaxFrom16bitLogits(const int16_t* logits, const size_t vocabSize) {
    // Comparisons is significantly faster in 32-bit,
    // so we store a temp 32-bit casted buffer and do comparisons in 32-bit.
    auto tmp = make32BitCopy(logits, vocabSize);
    return argmaxWithMax(tmp, vocabSize).first;
}

TokenType argmaxFrom16bitLogits(const __fp16* logits, const size_t vocabSize) {
    // Comparisons is significantly faster in 32-bit,
    // so we store a temp 32-bit casted buffer and do comparisons in 32-bit.
    auto tmp = make32BitCopy(logits, vocabSize);
    return argmaxWithMax(tmp, vocabSize).first;
}

void convertToSoftmax(float* array, const size_t vocabSize, const float max,
                      const float temperature) {
    if (temperature == 0) {
        const auto top1Token = argmaxWithMax(array, vocabSize).first;
        for (size_t j = 0; j < vocabSize; j++) {
            array[j] = 0;
        }
        array[top1Token] = 1;
    } else {
        // Apply softmax on array
        const float lowerBound = 1e-8;
        const float temperatureForSoftmax = std::max(lowerBound, temperature);
        float total = 0;
        for (size_t j = 0; j < vocabSize; j++) {
            array[j] = expf((array[j] - max) / temperatureForSoftmax);
            total += array[j];
        }
        for (size_t j = 0; j < vocabSize; j++) {
            array[j] /= total;
        }
    }
};

void convertToSoftmax(float* array, const size_t vocabSize, const float temperature) {
    const auto [argmax, max] = argmaxWithMax(array, vocabSize);

    if (temperature == 0) {
        for (size_t j = 0; j < vocabSize; j++) {
            array[j] = 0;
        }
        array[argmax] = 1;
        return;
    }

    // Apply softmax on array
    convertToSoftmax(array, vocabSize, max, temperature);
};

template <typename LogitsType>
void makeSoftmax(std::vector<float>& softmaxBuffer, const LogitsType* logits,
                 const size_t vocabSize, const float temperature, const float qscale) {
    softmaxBuffer.assign(logits, logits + vocabSize);

    // Dequant
    if (qscale != 1.0f) {
        for (auto& val : softmaxBuffer) {
            val *= qscale;
        }
    }
    convertToSoftmax(softmaxBuffer.data(), vocabSize, temperature);
}

void makeSoftmax(std::vector<float>& softmaxBuffer, const LLMType logitsType, const void* logits,
                 const size_t vocabSize, const float temperature, const float qscale) {
    switch (logitsType) {
        case LLMType::INT16:
            return makeSoftmax(softmaxBuffer, reinterpret_cast<const int16_t*>(logits), vocabSize,
                               temperature, qscale);
        case LLMType::FP16:
            return makeSoftmax(
                softmaxBuffer, reinterpret_cast<const __fp16*>(logits), vocabSize, temperature);

        case LLMType::FP32:
            return makeSoftmax(
                softmaxBuffer, reinterpret_cast<const float*>(logits), vocabSize, temperature);

        default:
            LOG(ERROR) << "makeSoftmax only supports INT16/FP16/FP32 logits.";
    }
}

ArgmaxProb argmaxProbFrom16bitLogits(const LLMType logitsType, const void* logits,
                                     const size_t vocabSize, const float modelOutputQuantScale) {
    // This function returns the highest probability and the corresponding index.
    switch (logitsType) {
        case LLMType::INT16:
            return argmaxProbFrom16bitLogits(
                reinterpret_cast<const int16_t*>(logits), vocabSize, modelOutputQuantScale);
        case LLMType::FP16:
            return argmaxProbFrom16bitLogits(reinterpret_cast<const __fp16*>(logits), vocabSize);
        default:
            LOG(ERROR) << "argmaxProbFrom16bitLogits function only supports INT16/FP16 logits.";
            return {0, 0};
    }
}

ArgmaxProb argmaxProbFrom16bitLogits(const int16_t* logits, const size_t vocabSize,
                                     const float modelOutputQuantScale) {
    // Comparisons is significantly faster in 32-bit,
    // so we store a temp 32-bit casted buffer and do comparisons in 32-bit.
    float total = 0;
    auto tmp = make32BitCopy(logits, vocabSize);
    const auto [index, max] = argmaxWithMax(tmp, vocabSize);
    for (size_t j = 0; j < vocabSize; j++) {
        total += expf((tmp[j] - max) * modelOutputQuantScale);
    }
    const float prob = 1 / total;
    return {index, prob};
}

ArgmaxProb argmaxProbFrom16bitLogits(const __fp16* logits, const size_t vocabSize) {
    // Comparisons is significantly faster in 32-bit,
    // so we store a temp 32-bit casted buffer and do comparisons in 32-bit.
    float total = 0;
    auto tmp = make32BitCopy(logits, vocabSize);
    const auto [index, max] = argmaxWithMax(tmp, vocabSize);
    for (size_t j = 0; j < vocabSize; j++) {
        total += expf(tmp[j] - max);
    }
    const float prob = 1 / total;
    return {index, prob};
}

ArgmaxProb randomSampleFrom16bitLogits(const LLMType logitsType, const void* logits,
                                       const size_t vocabSize, const float modelOutputQuantScale,
                                       const float temperature) {
    // This function returns the index that has highest probability and its probability.
    switch (logitsType) {
        case LLMType::INT16:
            return randomSampleFrom16bitLogits(reinterpret_cast<const int16_t*>(logits), vocabSize,
                                               modelOutputQuantScale, temperature);
        case LLMType::FP16:
            return randomSampleFrom16bitLogits(
                reinterpret_cast<const __fp16*>(logits), vocabSize, temperature);
        default:
            LOG(ERROR) << "randomSampleFrom16bitLogits function only supports INT16/FP16 logits.";
            return {0, 0};
    }
}

ArgmaxProb randomSampleFrom16bitLogits(const int16_t* logits, const size_t vocabSize,
                                       const float modelOutputQuantScale, const float temperature) {
    // Comparisons is significantly faster in 32-bit,
    // so we store a temp 32-bit casted buffer and do comparisons in 32-bit.
    if (temperature == 0) {
        auto tmp = make32BitCopy(logits, vocabSize);
        return {argmaxWithMax(tmp, vocabSize).first, 1};
    }

    float* distWithTemperature =
        makeTempSoftmax(logits, vocabSize, temperature, modelOutputQuantScale);

    // temperature sampling
    const size_t sampledToken = samplingFromDistribution(distWithTemperature, vocabSize);
    return {sampledToken, distWithTemperature[sampledToken]};
}

ArgmaxProb randomSampleFrom16bitLogits(const __fp16* logits, const size_t vocabSize,
                                       const float temperature) {
    // Comparisons is significantly faster in 32-bit,
    // so we store a temp 32-bit casted buffer and do comparisons in 32-bit.
    if (temperature == 0) {
        auto tmp = make32BitCopy(logits, vocabSize);
        return {argmaxWithMax(tmp, vocabSize).first, 1};
    }

    float* distWithTemperature = makeTempSoftmax(logits, vocabSize, temperature);

    // temperature sampling
    const size_t sampledToken = samplingFromDistribution(distWithTemperature, vocabSize);
    return {sampledToken, distWithTemperature[sampledToken]};
}

ArgmaxProb argmaxProbFrom16bitLogits(const LLMType logitsType, const void* logits,
                                     const size_t vocabSize, const float modelOutputQuantScale,
                                     const TokenType tokenIdForProb) {
    // This function returns the index that has highest probability and the probability of a given
    // token 'tokenIdForProb'.
    switch (logitsType) {
        case LLMType::INT16:
            return argmaxProbFrom16bitLogits(reinterpret_cast<const int16_t*>(logits), vocabSize,
                                             modelOutputQuantScale, tokenIdForProb);
        case LLMType::FP16:
            return argmaxProbFrom16bitLogits(
                reinterpret_cast<const __fp16*>(logits), vocabSize, tokenIdForProb);
        default:
            LOG(ERROR) << "argmaxProbFrom16bitLogits function only supports INT16/FP16 logits.";
            return {0, 0};
    }
}

ArgmaxProb argmaxProbFrom16bitLogits(const int16_t* logits, const size_t vocabSize,
                                     const float modelOutputQuantScale,
                                     const TokenType tokenIdForProb) {
    // Comparisons is significantly faster in 32-bit,
    // so we store a temp 32-bit casted buffer and do comparisons in 32-bit.
    float total = 0;
    auto tmp = make32BitCopy(logits, vocabSize);
    const auto [index, max] = argmaxWithMax(tmp, vocabSize);
    for (size_t j = 0; j < vocabSize; j++) {
        total += expf((tmp[j] - max) * modelOutputQuantScale);
    }
    const float prob = expf((tmp[tokenIdForProb] - max) * modelOutputQuantScale) / total;
    return {index, prob};
}

ArgmaxProb argmaxProbFrom16bitLogits(const __fp16* logits, const size_t vocabSize,
                                     const TokenType tokenIdForProb) {
    // Comparisons is significantly faster in 32-bit,
    // so we store a temp 32-bit casted buffer and do comparisons in 32-bit.
    float total = 0;
    auto tmp = make32BitCopy(logits, vocabSize);
    const auto [index, max] = argmaxWithMax(tmp, vocabSize);
    for (size_t j = 0; j < vocabSize; j++) {
        total += expf(tmp[j] - max);
    }
    const float prob = expf(tmp[tokenIdForProb] - max) / total;
    return {index, prob};
}

ArgmaxProb randomSampleFrom16bitLogits(const LLMType logitsType, const void* logits,
                                       const size_t vocabSize, const float modelOutputQuantScale,
                                       const float temperature, const TokenType tokenIdForProb) {
    // This function returns the index that has highest probability and the probability of a given
    // token 'tokenIdForProb'.
    switch (logitsType) {
        case LLMType::INT16:
            return randomSampleFrom16bitLogits(reinterpret_cast<const int16_t*>(logits), vocabSize,
                                               modelOutputQuantScale, temperature, tokenIdForProb);
        case LLMType::FP16:
            return randomSampleFrom16bitLogits(
                reinterpret_cast<const __fp16*>(logits), vocabSize, temperature, tokenIdForProb);
        default:
            LOG(ERROR) << "randomSampleFrom16bitLogits function only supports INT16/FP16 logits.";
            return {0, 0};
    }
}

ArgmaxProb randomSampleFrom16bitLogits(const int16_t* logits, const size_t vocabSize,
                                       const float modelOutputQuantScale, const float temperature,
                                       const TokenType tokenIdForProb) {
    if (temperature == 0) {
        auto tmp = make32BitCopy(logits, vocabSize);
        const auto top1Token = argmaxWithMax(tmp, vocabSize).first;
        return {top1Token, float(top1Token == tokenIdForProb)};
    }

    float* distWithTemperature =
        makeTempSoftmax(logits, vocabSize, temperature, modelOutputQuantScale);

    // temperature sampling
    const size_t sampledToken = samplingFromDistribution(distWithTemperature, vocabSize);
    return {sampledToken, distWithTemperature[tokenIdForProb]};
}

ArgmaxProb randomSampleFrom16bitLogits(const __fp16* logits, const size_t vocabSize,
                                       const float temperature, const TokenType tokenIdForProb) {
    if (temperature == 0) {
        auto tmp = make32BitCopy(logits, vocabSize);
        const auto top1Token = argmaxWithMax(tmp, vocabSize).first;
        return {top1Token, float(top1Token == tokenIdForProb)};
    }

    float* distWithTemperature = makeTempSoftmax(logits, vocabSize, temperature);

    // temperature sampling
    const size_t sampledToken = samplingFromDistribution(distWithTemperature, vocabSize);
    return {sampledToken, distWithTemperature[tokenIdForProb]};
}

ArgmaxProbs randomSampleFrom16bitLogits(const LLMType logitsType, const void* logits,
                                        const size_t vocabSize, const float modelOutputQuantScale,
                                        const float temperature,
                                        const std::vector<TokenType>& tokensIdForProb) {
    // This function returns the index that has highest probability and the probability of a given
    // token 'tokenIdForProb'.
    switch (logitsType) {
        case LLMType::INT16:
            return randomSampleFrom16bitLogits(reinterpret_cast<const int16_t*>(logits), vocabSize,
                                               modelOutputQuantScale, temperature, tokensIdForProb);
        case LLMType::FP16:
            return randomSampleFrom16bitLogits(
                reinterpret_cast<const __fp16*>(logits), vocabSize, temperature, tokensIdForProb);
        default:
            LOG(ERROR) << "randomSampleFrom16bitLogits function only supports INT16/FP16 logits.";
            return {0, {0}};
    }
}

ArgmaxProbs randomSampleFrom16bitLogits(const int16_t* logits, const size_t vocabSize,
                                        const float modelOutputQuantScale, const float temperature,
                                        const std::vector<TokenType>& tokensIdForProb) {
    if (temperature == 0) {
        auto tmp = make32BitCopy(logits, vocabSize);
        const auto top1Token = argmaxWithMax(tmp, vocabSize).first;
        std::vector<float> probs;
        for (auto token : tokensIdForProb) {
            probs.push_back(float(top1Token == token));
        }
        return {top1Token, probs};
    }

    float* distWithTemperature =
        makeTempSoftmax(logits, vocabSize, temperature, modelOutputQuantScale);

    // temperature sampling
    const size_t sampledToken = samplingFromDistribution(distWithTemperature, vocabSize);
    auto probs = gather(distWithTemperature, tokensIdForProb, vocabSize);
    return {sampledToken, probs};
}

ArgmaxProbs randomSampleFrom16bitLogits(const __fp16* logits, const size_t vocabSize,
                                        const float temperature,
                                        const std::vector<TokenType>& tokensIdForProb) {
    if (temperature == 0) {
        auto tmp = make32BitCopy(logits, vocabSize);
        const auto top1Token = argmaxWithMax(tmp, vocabSize).first;
        std::vector<float> probs;
        for (auto token : tokensIdForProb) {
            probs.push_back(float(top1Token == token));
        }
        return {top1Token, probs};
    }

    float* distWithTemperature = makeTempSoftmax(logits, vocabSize, temperature);

    // temperature sampling
    const size_t sampledToken = samplingFromDistribution(distWithTemperature, vocabSize);
    auto probs = gather(distWithTemperature, tokensIdForProb, vocabSize);
    return {sampledToken, probs};
}

TokenType argmaxFromAdjustDistSpecDec(const LLMType logitsType, const void* targetLogits,
                                      const void* draftLogits, const size_t vocabSize,
                                      const float targetOutputQuantScale,
                                      const float draftOutputQuantScale) {
    switch (logitsType) {
        case LLMType::INT16:
            return argmaxFromAdjustDistSpecDec(reinterpret_cast<const int16_t*>(targetLogits),
                                               reinterpret_cast<const int16_t*>(draftLogits),
                                               vocabSize, targetOutputQuantScale,
                                               draftOutputQuantScale);
        case LLMType::FP16:
            return argmaxFromAdjustDistSpecDec(reinterpret_cast<const __fp16*>(targetLogits),
                                               reinterpret_cast<const __fp16*>(draftLogits),
                                               vocabSize, targetOutputQuantScale,
                                               draftOutputQuantScale);
        default:
            LOG(ERROR) << "argmaxFromAdjustDistSpecDec function only supports INT16/FP16 logits.";
            return 0;
    }
}

template <typename LogitsType>
TokenType argmaxFromAdjustDistSpecDec(const LogitsType* targetLogits, const LogitsType* draftLogits,
                                      const size_t vocabSize, const float targetOutputQuantScale,
                                      const float draftOutputQuantScale) {
    const float temperature = 0;
    float* targetDist =
        makeTempSoftmax<0>(targetLogits, vocabSize, temperature, targetOutputQuantScale);
    float* draftDist =
        makeTempSoftmax<1>(draftLogits, vocabSize, temperature, draftOutputQuantScale);

    // Adjust the distribution
    // p'(x) = max(p(x)-q(x), 0)
    for (size_t i = 0; i < vocabSize; i++) {
        targetDist[i] = std::max(targetDist[i] - draftDist[i], 0.0f);
    }
    // argmax(p'(x))
    return argmaxWithMax(targetDist, vocabSize).first;
}

TokenType randomSampleFromAdjustDistSpecDec(const LLMType logitsType, const void* targetLogits,
                                            const void* draftLogits, const size_t vocabSize,
                                            const float targetOutputQuantScale,
                                            const float draftOutputQuantScale,
                                            const float targetSamplingTemperature,
                                            const float draftSamplingTemperature) {
    switch (logitsType) {
        case LLMType::INT16:
            return randomSampleFromAdjustDistSpecDec(
                reinterpret_cast<const int16_t*>(targetLogits),
                reinterpret_cast<const int16_t*>(draftLogits), vocabSize, targetOutputQuantScale,
                draftOutputQuantScale, targetSamplingTemperature, draftSamplingTemperature);
        case LLMType::FP16:
            return randomSampleFromAdjustDistSpecDec(
                reinterpret_cast<const __fp16*>(targetLogits),
                reinterpret_cast<const __fp16*>(draftLogits), vocabSize, targetOutputQuantScale,
                draftOutputQuantScale, targetSamplingTemperature, draftSamplingTemperature);
        default:
            LOG(ERROR)
                << "randomSampleFromAdjustDistSpecDec function only supports INT16/FP16 logits.";
            return 0;
    }
}

template <typename LogitsType>
TokenType randomSampleFromAdjustDistSpecDec(const LogitsType* targetLogits,
                                            const LogitsType* draftLogits, const size_t vocabSize,
                                            const float targetOutputQuantScale,
                                            const float draftOutputQuantScale,
                                            const float targetSamplingTemperature,
                                            const float draftSamplingTemperature) {
    // Large model
    float* targetDist = makeTempSoftmax<0>(
        targetLogits, vocabSize, targetSamplingTemperature, targetOutputQuantScale);
    // Small model
    float* draftDist =
        makeTempSoftmax<1>(draftLogits, vocabSize, draftSamplingTemperature, draftOutputQuantScale);

    // Adjust the distribution
    // p'(x) = max(p(x)-q(x), 0)
    float total = 0;
    for (size_t i = 0; i < vocabSize; i++) {
        targetDist[i] = std::max(targetDist[i] - draftDist[i], 0.0f);
        total += targetDist[i];
    }
    for (size_t i = 0; i < vocabSize; i++) {
        targetDist[i] /= total;
    }
    // sample from (p'(x))
    const size_t sampledToken = samplingFromDistribution(targetDist, vocabSize);
    return sampledToken;
}

template <typename LogitsType>
inline std::vector<TokenType> getTopkArgmaxV2For32Bit(LogitsType* logits, const size_t vocabSize,
                                                      const size_t k) {
    static_assert(sizeof(LogitsType) == 4);
    std::vector<TokenType> result(k);
    TokenType topi;
    for (size_t i = 0; i < k; i++) {
        if (i > 0) {
            logits[topi] = std::numeric_limits<LogitsType>::min();
        }
        topi = argmaxWithMax(logits, vocabSize).first;
        result[i] = topi;
    }
    return result;
}

std::vector<TokenType> getTopkArgmaxV2(const LLMType logitsType, const void* logits,
                                       const size_t vocabSize, const size_t k) {
    auto tmp = make32BitCopy(logitsType, logits, vocabSize);
    if (isFloating(logitsType))
        return getTopkArgmaxV2For32Bit(reinterpret_cast<float*>(tmp), vocabSize, k);
    else
        return getTopkArgmaxV2For32Bit(reinterpret_cast<int*>(tmp), vocabSize, k);
}

template <typename LogitsType>
std::pair<std::vector<TokenType>, std::vector<float>>
getTopkArgmaxAndLogitsV2For32Bit(LogitsType* logits, const size_t vocabSize, const size_t k,
                                 const float modelOutputQuantScale, const float temperature) {
    static_assert(sizeof(LogitsType) == 4);
    auto probs = makeTempSoftmax(logits, vocabSize, temperature, modelOutputQuantScale);
    auto cmp = [](const auto& left, const auto& right) { return left.second > right.second; };

    // clang-format off
    std::priority_queue<std::pair<TokenType, float>,
                        std::vector<std::pair<TokenType, float>>,
                        decltype(cmp)> minHeap(cmp);
    // clang-format on

    for (size_t i = 0; i < vocabSize; i++) {
        if (minHeap.size() < k || probs[i] > minHeap.top().second) {
            if (minHeap.size() == k) {
                minHeap.pop();
            }
            minHeap.emplace(i, probs[i]);
        }
    }
    std::vector<TokenType> result(k);
    std::vector<float> topKlogits(k);
    for (size_t i = 0; i < k; i++) {
        if (minHeap.empty())
            break;
        auto& top = minHeap.top();
        result[k - i - 1] = top.first;
        topKlogits[k - i - 1] = top.second;
        minHeap.pop();
    }
    return {result, topKlogits};
}

std::pair<std::vector<TokenType>, std::vector<float>>
getTopkArgmaxAndLogitsV2(const LLMType logitsType, const void* logits, const size_t vocabSize,
                         const size_t k, const float modelOutputQuantScale,
                         const float temperature) {
    if (temperature == 0) {
        auto topKTokens = getTopkArgmaxV2(logitsType, logits, vocabSize, k);
        std::vector<float> topKProbs = {1.0f};
        topKProbs.resize(k, 0.0f);
        return {topKTokens, topKProbs};
    }
    auto tmp = make32BitCopy(logitsType, logits, vocabSize);
    if (isFloating(logitsType))
        return getTopkArgmaxAndLogitsV2For32Bit(
            reinterpret_cast<float*>(tmp), vocabSize, k, 1.0, temperature);
    else
        return getTopkArgmaxAndLogitsV2For32Bit(
            reinterpret_cast<int*>(tmp), vocabSize, k, modelOutputQuantScale, temperature);
}

template <typename LogitsType>
std::vector<TokenType> getTopkArgmax(const LogitsType* logits, const size_t vocabSize,
                                     const size_t k) {
    using ValIdxPair = std::pair<LogitsType, size_t>;
    // clang-format off
    std::priority_queue<ValIdxPair,
                        std::vector<ValIdxPair>,
                        std::greater<ValIdxPair>> q;
    // clang-format on
    for (size_t i = 0; i < vocabSize; ++i) {
        if (q.size() < k)
            q.push(ValIdxPair(logits[i], i));
        else if (q.top().first < logits[i]) {
            q.pop();
            q.push(ValIdxPair(logits[i], i));
        }
    }
    std::vector<TokenType> result(k);
    for (size_t i = 0; i < k; ++i) {
        result[k - i - 1] = static_cast<TokenType>(q.top().second);
        q.pop();
    }
    return result;
}

std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>>
conductTreeMask(const std::vector<int>& parentId) {
    const size_t size = parentId.size();
    std::vector<std::vector<int>> treeMask(size, std::vector<int>(size, 0));
    std::vector<bool> alreadySearched(size, 0);
    std::vector<std::vector<int>> allCandidates;
    for (int i = size - 1; i >= 0; i--) {
        std::vector<int> candidate;
        bool overlap = false;
        treeMask[i][i] = 1;
        int x = i;
        if (alreadySearched[x]) {
            // Do not record the path
            overlap = true;
        } else {
            candidate.push_back(x);
        }
        while (parentId[x] != -1) {
            // Record path if this path is non-overlapped
            if (!overlap)
                candidate.push_back(parentId[x]);
            // Mark node true
            alreadySearched[parentId[x]] = true;
            // Fill val
            const auto parent = parentId[x];
            treeMask[i][parent] = 1;
            x = parent;
        }
        if (!overlap) {
            std::reverse(candidate.begin(), candidate.end());
            allCandidates.push_back(candidate);
        }
    }
    std::sort(allCandidates.begin(), allCandidates.end());
    return {treeMask, allCandidates};
}

std::vector<std::vector<int>> horizontalConcat(const std::vector<std::vector<int>>& leftMask,
                                               const std::vector<std::vector<int>>& rightMask) {
    if (leftMask.empty())
        return rightMask;
    std::vector<std::vector<int>> mask;
    const size_t height = leftMask.size();
    DCHECK_EQ(leftMask.size(), rightMask.size());
    const size_t leftWidth = leftMask[0].size();
    const size_t rightWidth = rightMask[0].size();
    for (size_t i = 0; i < height; i++) {
        std::vector<int> newRowVector;
        for (size_t j = 0; j < leftWidth; j++) {
            newRowVector.push_back(leftMask[i][j]);
        }
        for (size_t k = 0; k < rightWidth; k++) {
            newRowVector.push_back(rightMask[i][k]);
        }
        mask.push_back(newRowVector);
    }
    return mask;
}

// Returns true if preformatter has been successfully added, false if otherwise.
bool addPreformatter(const std::string& prefName, std::string& prompt) {
    if (prefName.empty())
        return false;

#define DISPATCH(NAME)                           \
    if (prefName == #NAME) {                     \
        prompt = addPreformatter_##NAME(prompt); \
        return true;                             \
    }
    DISPATCH(AlpacaNoInput)
    DISPATCH(OneShotConversation)
    DISPATCH(VicunaNoInput)
    DISPATCH(QwenNoInput)
    DISPATCH(Qwen3NoInput)
    DISPATCH(Qwen3NoInputNoThink)
    DISPATCH(Qwen3VLNoInput)
    DISPATCH(Llama3NoInput)
    DISPATCH(Phi3NoInput)
    DISPATCH(MinicpmNoInput)
    DISPATCH(MinicpmNoInputZh)
    DISPATCH(InternLM2)
    DISPATCH(GemmaNoInput)
#undef DISPATCH
    return false;
}

std::string addPreformatter_AlpacaNoInput(const std::string& prompt) {
    std::stringstream ss;
    ss << "Below is an instruction that describes a task. Write a response that appropriately "
          "completes the request.\n\n### Instruction:\n"
       << prompt << "\n\n### Response:\n";
    return ss.str();
}

std::string addPreformatter_OneShotConversation(const std::string& prompt) {
    std::stringstream ss;
    ss << "A chat between a curious human and an artificial intelligence assistant. "
          "The assistant gives helpful, detailed, and polite answers to the human's questions.\n"
          "### Human: Got any creative ideas for a 10 year old’s birthday?\n### Assistant: "
          "Of course! Here are some creative ideas for a 10-year-old's birthday party:\n"
          "1. Treasure Hunt: Organize a treasure hunt in your backyard or nearby park. Create "
          "clues and riddles for the kids to solve, leading them to hidden treasures and "
          "surprises.\n"
          "2. Science Party: Plan a science-themed party where kids can engage in fun and "
          "interactive experiments. You can set up different stations with activities like making "
          "slime, erupting volcanoes, or creating simple chemical reactions.\n"
          "3. Outdoor Movie Night: Set up a backyard movie night with a projector and a large "
          "screen or white sheet. Create a cozy seating area with blankets and pillows, and serve "
          "popcorn and snacks while the kids enjoy a favorite movie under the stars.\n"
          "Remember to tailor the activities to the birthday child's interests and preferences. "
          "Have a great celebration!\n### Human: "
       << prompt << "\n### Assistant:";
    return ss.str();
}

std::string addPreformatter_VicunaNoInput(const std::string& prompt) {
    std::stringstream ss;
    ss << "A chat between a curious user and an artificial intelligence assistant. "
          "The assistant gives helpful, detailed, and polite answers to the user's questions. "
          "USER: "
       << prompt << " ASSISTANT:";
    return ss.str();
}

std::string addPreformatter_QwenNoInput(const std::string& prompt) {
    std::stringstream ss;
    ss << "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
       << prompt << "<|im_end|>\n<|im_start|>assistant\n";
    return ss.str();
}

std::string addPreformatter_Qwen3NoInputNoThink(const std::string& prompt) {
    std::stringstream ss;
    ss << "<|im_start|>user\n"
       << prompt << "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    return ss.str();
}

std::string addPreformatter_Qwen3NoInput(const std::string& prompt) {
    std::stringstream ss;
    ss << "<|im_start|>user\n" << prompt << "<|im_end|>\n<|im_start|>assistant\n<think>\n";
    return ss.str();
}

// Qwen3-VL chat template, aligned with HF/Python qwen3_preformatter.json:
//   "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
// 区别于 Qwen3NoInput*: 无 system 段，无 <think>...</think> 块。配合 --image-style qwen3vl
// 把 prompt 中的 "<image>" 替换为 "<|vision_start|><image><|vision_end|>"，构成与
// Qwen3-VL 训练时完全一致的 chat 模板。用于端侧 mmbench / VL 任务，不要用于纯文本 Qwen3。
std::string addPreformatter_Qwen3VLNoInput(const std::string& prompt) {
    std::stringstream ss;
    ss << "<|im_start|>user\n" << prompt << "<|im_end|>\n<|im_start|>assistant\n";
    return ss.str();
}

std::string applyImageStyle(const std::string& prompt, const std::string& style) {
    if (style != "qwen3vl") {
        return prompt; // "bare" 或未识别 style：原样返回
    }
    static const std::string kFrom = "<image>";
    static const std::string kTo = "<|vision_start|><image><|vision_end|>";
    std::string out;
    out.reserve(prompt.size() + 64);
    size_t pos = 0;
    while (true) {
        size_t found = prompt.find(kFrom, pos);
        if (found == std::string::npos) {
            out.append(prompt, pos, std::string::npos);
            break;
        }
        out.append(prompt, pos, found - pos);
        out.append(kTo);
        pos = found + kFrom.size();
    }
    return out;
}

std::string defaultImageStyleFor(const std::string& preformatterName) {
    if (preformatterName == "Qwen3VLNoInput") {
        return "qwen3vl";
    }
    return "bare";
}

std::string addPreformatter_Llama3NoInput(const std::string& prompt) {
    std::stringstream ss;
    ss << "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
       << prompt << "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    return ss.str();
}

std::string addPreformatter_Phi3NoInput(const std::string& prompt) {
    std::stringstream ss;
    ss << "<|system|>\nYou are a helpful AI assistant. Please provide safe, ethical and accurate "
          "information to the user.\n<|user|>\n "
       << prompt << " \n <|assistant|>";
    return ss.str();
}

std::string addPreformatter_MinicpmNoInput(const std::string& prompt) {
    std::stringstream ss;
    ss << "<USER>" << prompt << "<AI>";
    return ss.str();
}

std::string addPreformatter_MinicpmNoInputZh(const std::string& prompt) {
    std::stringstream ss;
    ss << "<用戶>" << prompt << "<AI>";
    return ss.str();
}

std::string addPreformatter_InternLM2(const std::string& prompt) {
    std::stringstream ss;
    ss << "<|im_start|>system\nYou are an AI assistant whose name is InternLM (书生·浦语)."
       << "\n- InternLM (书生·浦语) is a conversational language model that is developed by "
          "Shanghai AI Laboratory (上海人工智能实验室)."
       << " It is designed to be helpful, honest, and harmless.\n"
       << "- InternLM (书生·浦语) can understand and communicate fluently in the language chosen "
          "by the user such as English and 中文."
       << "<|im_end|>\n<|im_start|>user\n"
       << prompt << "<|im_end|>\n<|im_start|>assistant\n";
    return ss.str();
}

std::string addPreformatter_GemmaNoInput(const std::string& prompt) {
    std::stringstream ss;
    ss << "<start_of_turn>user\n" << prompt << "<|end_of_turn>\n<start_of_turn>model\n";
    return ss.str();
}

std::vector<std::string> split(const std::string& str, const std::string& sep) {
    std::vector<std::string> substrings;
    const std::regex noSepPattern("([^" + sep + "]+)");
    std::smatch match;
    auto cur = str.cbegin();
    while (std::regex_search(cur, str.cend(), match, noSepPattern)) {
        substrings.push_back(match[0].str());
        cur = match.suffix().first;
    }
    return substrings;
}

// Parses non-digit separated integers (the separator can be anything other than numbers)
std::vector<TokenType> parseTokenString(const std::string& tokenString) {
    std::vector<TokenType> tokens;
    const std::regex tokenPattern("([0-9-]+)"); // Matches a signed integer
    std::smatch match;
    auto cur = tokenString.cbegin();
    while (std::regex_search(cur, tokenString.cend(), match, tokenPattern)) {
        tokens.push_back(std::stoi(match[0].str()));
        cur = match.suffix().first;
    }
    return tokens;
}

std::vector<std::string> readPromptFiles(const std::vector<std::string>& promptPaths,
                                         const bool onePromptPerLine) {
    std::vector<std::string> prompts;

    // Replace "\\n" with '\n'
    auto replaceNewLines = [&](std::string& str) -> decltype(auto) {
        constexpr std::string_view literalNewLine = "\\n";
        constexpr std::string_view realNewLine = "\n";
        size_t start_pos = 0;
        while ((start_pos = str.find(literalNewLine, start_pos)) != std::string::npos) {
            str.replace(start_pos, literalNewLine.size(), realNewLine);
            start_pos += realNewLine.size();
        }
        return str;
    };

    auto readLines = [&](std::ifstream& fin) {
        std::string prompt;
        while (std::getline(fin, prompt) && !prompt.empty()) {
            if (isWhiteLine(prompt))
                continue;
            prompts.emplace_back(std::move(replaceNewLines(prompt)));
        }
    };

    auto readFile = [&](std::ifstream& fin) {
        std::stringstream buffer;
        buffer << fin.rdbuf();
        prompts.push_back(buffer.str());
    };

    for (const auto& path : promptPaths) {
        std::ifstream fin(path);
        if (!fin) {
            LOG(ERROR) << "Unable to open the prompt file: " << path;
            continue;
        } else {
            LOG(INFO) << "Reading prompt from file: " << path;
        }
        if (onePromptPerLine) {
            readLines(fin);
        } else {
            readFile(fin);
        }
    }
    return prompts;
}

// Explicit instantiation of getTopkArgmax for some logits types
template std::vector<TokenType> getTopkArgmax<int16_t>(const int16_t* logits,
                                                       const size_t vocabSize, const size_t k);
template std::vector<TokenType> getTopkArgmax<__fp16>(const __fp16* logits, const size_t vocabSize,
                                                      const size_t k);
template std::vector<TokenType> getTopkArgmax<float>(const float* logits, const size_t vocabSize,
                                                     const size_t k);

} // namespace utils