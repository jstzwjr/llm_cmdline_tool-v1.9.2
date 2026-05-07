#include "config_parser.h"

#include "common/file_source.h"
#include "common/logging.h"
#include "mtk_llm_options.h"
#include "mtk_llm_types.h"
#include "third_party/include/yaml-cpp/yaml.h"

#include <filesystem>
#include <regex>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace utils {

using TokenType = mtk::Tokenizer::TokenType;

void YamlConfigParser::parse(LlmModelOptions& modelOptions,
                             LlmRuntimeOptions& runtimeOptions) const {
    parseModelOptions(modelOptions);
    parseRuntimeOptions(runtimeOptions);
    parseLlmModelPaths(modelOptions, runtimeOptions);
}

void YamlConfigParser::parseModelOptions(LlmModelOptions& modelOptions) const {
    const auto& modelOptYaml = getModelOptionsYaml();

    // Validation
    if (!hasValue(modelOptYaml)) {
        LOG(FATAL) << "Invalid yaml config file: 'modelOptions' is not found in the config.";
    }

    // Parse model options
#define PARSE_OPTION(type, key)                           \
    if (modelOptYaml[#key]) {                             \
        modelOptions.key = modelOptYaml[#key].as<type>(); \
    }
    PARSE_OPTION(size_t, genModelBatchSize)
    PARSE_OPTION(size_t, promptTokenBatchSize)
    PARSE_OPTION(size_t, genTokenBatchSize)
    PARSE_OPTION(size_t, cacheSize)
    PARSE_OPTION(size_t, hiddenSize)
    PARSE_OPTION(size_t, numHead)
    PARSE_OPTION(size_t, numLayer)
    PARSE_OPTION(size_t, headDim)
    PARSE_OPTION(size_t, maxTokenLength)
    PARSE_OPTION(size_t, numMedusaHeads)
    PARSE_OPTION(float, rotEmbBase)
    PARSE_OPTION(float, ntkScale)
    PARSE_OPTION(size_t, rotEmbNumInputs)
    PARSE_OPTION(size_t, numAttnOutputs)
    PARSE_OPTION(float, embOutputQuantScale)
    PARSE_OPTION(float, modelOutputQuantScale)
    PARSE_OPTION(bool, splitMask)
#undef PARSE_OPTION

#define PARSE_OPTION_LLMTYPE(key)                                                                 \
    if (modelOptYaml[#key]) {                                                                     \
        modelOptions.key = mtk::getLLMTypeFromName(modelOptYaml[#key].as<std::string>().c_str()); \
    }
    PARSE_OPTION_LLMTYPE(modelInputType)
    PARSE_OPTION_LLMTYPE(modelOutputType)
    PARSE_OPTION_LLMTYPE(cacheType)
    PARSE_OPTION_LLMTYPE(maskType)
    PARSE_OPTION_LLMTYPE(rotEmbType)
#undef PARSE_OPTION_LLMTYPE

    // Calculate headDim if not provided
    if (modelOptions.headDim == 0) {
        modelOptions.headDim = modelOptions.hiddenSize / modelOptions.numHead;
        LOG(DEBUG) << "ModelOptions: headDim value calculated as hiddenSize / numHead: "
                   << modelOptions.headDim;
    }

    // Deprecate 'modelBatchSize' and use 'genModelBatchSize' instead
    if (modelOptYaml["modelBatchSize"]) {
        LOG(WARN) << "The model option 'modelBatchSize' is deprecated. Its value will be assigned "
                     "to 'genModelBatchSize'.";
        modelOptions.genModelBatchSize = modelOptYaml["modelBatchSize"].as<size_t>();
    }

    if (modelOptions.embOutputQuantScale != 0) {
        // TODO: Deprecated
        LOG(WARN) << "The use of 'embOutputQuantScale' is deprecated. Please ensure the token "
                     "embedding Lut value type matches with the model embedding input type.";
    }

    const auto numAttnOutputs = modelOptions.numAttnOutputs;
    if (numAttnOutputs != 0 && numAttnOutputs != 1 && numAttnOutputs != modelOptions.numLayer) {
        LOG(WARN) << "The model option 'numAttnOutputs' must be equal to 0, 1, or numLayer ("
                  << modelOptions.numLayer << "), but got " << numAttnOutputs << " instead.";
    }

    const auto outputType = modelOptYaml["modelOutputType"].as<std::string>();
    const auto outputScale = modelOptions.modelOutputQuantScale;
    if (outputType == "FP16" && outputScale != 1.0) {
        modelOptions.modelOutputQuantScale = 1.0;
        LOG(WARN) << "Overriding scale to 1.0 for FP16 output.";
    }
}

void YamlConfigParser::parseRuntimeOptions(LlmRuntimeOptions& runtimeOptions) const {
    const auto& runtimeOptYaml = getRuntimeOptionsYaml();

    const auto& specialTokensYaml = runtimeOptYaml["specialTokens"];
    const auto& tokenizerRegexYaml = runtimeOptYaml["tokenizerRegex"];
    const auto& vocabPathYaml = runtimeOptYaml["vocabPath"]; // TODO: Deprecated
    const auto& tokenizerPathYaml = runtimeOptYaml["tokenizerPath"];
    const auto& tfliteEmbPathYaml = runtimeOptYaml["tfliteEmbPath"]; // TODO: Deprecated
    const auto& tokenEmbPathYaml = runtimeOptYaml["tokenEmbPath"];
    const auto& dlaLmHeadPathYaml = runtimeOptYaml["dlaLmHeadPath"];
    const auto& dlaMedusaHeadsPathYaml = runtimeOptYaml["dlaMedusaHeadsPath"];
    const auto& cachePathsYaml = runtimeOptYaml["cachePaths"];
    const auto& sharedWeightsPathsYaml = runtimeOptYaml["sharedWeightsPaths"];
    const auto& useLmHeadSharedWeightsYaml = runtimeOptYaml["useLmHeadSharedWeights"];
    const auto& loraWeightsPathsYaml = runtimeOptYaml["loraWeightsPaths"];
    const auto& initWithLoraKeyYaml = runtimeOptYaml["initWithLoraKey"];
    const auto& loraInputCountYaml = runtimeOptYaml["loraInputCount"];
    const auto& cacheEvictOptYaml = runtimeOptYaml["cacheEvictionOptions"];
    const size_t numCachePaths = cachePathsYaml ? cachePathsYaml.size() : 0;

    // Helper to get number of chunks
    auto getNumChunks = [&](const auto& key) {
        const auto& yamlNode = runtimeOptYaml[key];
        std::unordered_set<size_t> numChunkSet;
        for (const auto& kv : yamlNode) {
            const auto& paths = kv.second;
            if (!paths.IsNull())
                numChunkSet.insert(paths.size());
        }
        CHECK_LE(numChunkSet.size(), 1)
            << "Invalid yaml config file: Inconsistent chunk size for '" << key << "'";
        return numChunkSet.empty() ? 0 : *numChunkSet.cbegin();
    };

    // Basic prompt & gen model config. For backward compatibility.
    const auto& dlaPromptPathsYaml = runtimeOptYaml["dlaPromptPaths"];
    const auto& dlaGenPathsYaml = runtimeOptYaml["dlaGenPaths"];
    const size_t numPromptDla = dlaPromptPathsYaml ? dlaPromptPathsYaml.size() : 0;
    const size_t numGenDla = dlaGenPathsYaml ? dlaGenPathsYaml.size() : 0;

    // Advanced model config
    const auto& dlaPathsYaml = runtimeOptYaml["dlaPaths"];

    const size_t numDlaChunks =
        std::max(std::max(numPromptDla, numGenDla), getNumChunks("dlaPaths"));

    // Validation:
    //   - Both 'modelOptions' and 'runtimeOptions' have to be defined.
    //   - 'tokenizerPath' has to be defined. Otherwise will look for 'vocabPath' (deprecated).
    //   - 'runtimeOptions.tokenEmbPath' has to be defined.
    //   - At least one of 'dlaPromptPathsYaml' and 'dlaGenPathsYaml', or 'dlaPaths' has to be
    //   defined.
    //   - The number of cache paths (if provided) must match the dla chunk count.
    //   - The number of paths per LoRA (if provided) must match the dla chunk count.
    if (!runtimeOptYaml) {
        LOG(FATAL) << "Invalid yaml config file: 'runtimeOptions' is not found in the config.";
    }
    if (!tokenizerPathYaml && !vocabPathYaml) {
        LOG(FATAL) << "Invalid yaml config file: 'tokenizerPath' is not defined in the yaml "
                      "config.";
    }
    if (!tokenEmbPathYaml && !tfliteEmbPathYaml) {
        LOG(FATAL) << "Invalid yaml config file: 'tokenEmbPath' is not defined in the yaml config";
    }
    if (!numDlaChunks) {
        LOG(FATAL) << "Invalid yaml config file: At least one of 'dlaPromptPaths', 'dlaGenPaths', "
                   << "or 'dlaPaths' has to be defined in the yaml config.";
    }
    if (numCachePaths > 0 && numCachePaths != numDlaChunks) {
        LOG(FATAL) << "Invalid yaml config file: The number of provided cache paths ("
                   << numCachePaths << ") does not " << "match the number of dla chunks ("
                   << numDlaChunks << ").";
    }

    const size_t numLoraWeightsPaths = getNumChunks("loraWeightsPaths");
    if (numLoraWeightsPaths > 0 && numLoraWeightsPaths != numDlaChunks) {
        LOG(FATAL) << "Invalid yaml config file: The number of provided LoRA weights paths"
                   << "(" << numLoraWeightsPaths << ") does not " << "match the number of dla "
                   << "chunks " << numDlaChunks << ").";
    }

    // Parse runtime options
    runtimeOptions.startTokenIndex = tryParse<size_t>(runtimeOptYaml["startTokenIndex"]);

    // Token embedding bin
    if (!tokenEmbPathYaml) {
        // TODO: Deprecated
        LOG(WARN) << "The use of 'tfliteEmbPath' in YAML config is deprecated. "
                     "Please rename it to 'tokenEmbPath' instead.";
        runtimeOptions.tokenEmbFile = tfliteEmbPathYaml.as<std::string>();
        if (fs::path(tfliteEmbPathYaml.as<std::string>()).extension() == ".tflite") {
            LOG(ERROR) << "Token embedding file has '.tflite' extension. "
                          "Please note that '.tflite' embedding has been replaced with '.bin' "
                          "lookup table.";
        }
    } else {
        runtimeOptions.tokenEmbFile = tokenEmbPathYaml.as<std::string>();
    }

    // Tokenizer
    if (!tokenizerPathYaml && vocabPathYaml) {
        // TODO: Deprecated
        LOG(WARN) << "The use of 'vocabPath' in YAML config is deprecated. "
                     "Please use 'tokenizerPath' instead.";
        runtimeOptions.tokenizerPath = parseScalarOrSeq(vocabPathYaml);
    } else {
        runtimeOptions.tokenizerPath = parseScalarOrSeq(tokenizerPathYaml);
    }
    runtimeOptions.tokenizerRegex = tryParse<std::string>(tokenizerRegexYaml);

    // Special tokens:
    //   - Both 'bosId' and 'eosId' have to be defined.
    //   - 'addBos' is default to false, unless set to true in config.
    //   - if 'stopToken' is not defined, then it is default to eosId.
    if (!hasValue(specialTokensYaml)) {
        LOG(FATAL) << "The runtime option 'specialTokens' is required.";
    }
    const auto& bosIdYaml = specialTokensYaml["bosId"];
    const auto& eosIdYaml = specialTokensYaml["eosId"];
    const auto& addBosYaml = specialTokensYaml["addBos"];
    const auto& stopTokenYaml = specialTokensYaml["stopToken"];
    if (!hasValue(bosIdYaml) || !hasValue(eosIdYaml)) {
        LOG(FATAL) << "Both 'bosId' & 'eosId' special tokens have to be defined in the config.";
    } else {
        auto& specialTokens = runtimeOptions.specialTokens;
        specialTokens.bosId = bosIdYaml.as<TokenType>();
        specialTokens.eosId = eosIdYaml.as<TokenType>();
        specialTokens.addBos = tryParse<bool>(addBosYaml, false);
        // Stop token set
        if (!hasValue(stopTokenYaml)) {
            specialTokens.stopToken = {specialTokens.eosId};
            LOG(DEBUG) << "The option 'stopToken' is not specified, defaulting to EoS token: "
                       << specialTokens.eosId;
        } else if (stopTokenYaml.IsSequence()) {
            const auto& stopTokenVec = stopTokenYaml.as<std::vector<TokenType>>();
            specialTokens.stopToken = {stopTokenVec.begin(), stopTokenVec.end()};
        } else {
            specialTokens.stopToken = {stopTokenYaml.as<TokenType>()};
        }
    }

    // Optional separate LmHead (split-tail)
    runtimeOptions.dlaLmHeadFile = tryParse<std::string>(dlaLmHeadPathYaml);
    runtimeOptions.useLmHeadSharedWeights = tryParse<bool>(useLmHeadSharedWeightsYaml, false);

    // Optional medusa heads
    runtimeOptions.dlaMedusaHeadsFile = tryParse<std::string>(dlaMedusaHeadsPathYaml);

    // LoRA weights path
    for (const auto& kv : loraWeightsPathsYaml) {
        const auto& loraKey = kv.first.as<std::string>();
        const auto& weightsPaths = parseScalarOrSeq(kv.second);
        if (weightsPaths.empty())
            continue;
        runtimeOptions.loraWeightsFiles.emplace(loraKey, pathsToFileSources(weightsPaths));
    }

    // The default value of `runtimeOptions.initWithLoraKey` is an empty string.
    runtimeOptions.initWithLoraKey = tryParse<std::string>(initWithLoraKeyYaml);

    // The default value of `runtimeOptions.loraInputCount` is 0.
    runtimeOptions.loraInputCount = tryParse<size_t>(loraInputCountYaml);

    // Cache path
    runtimeOptions.cacheFiles = pathsToFileSources(parseScalarOrSeq(cachePathsYaml));

    // Optional NeuroPilot backend runtime library path
    runtimeOptions.npLibPath = tryParse<std::string>(runtimeOptYaml["npLibPath"]);

    // Shared weights path
    runtimeOptions.sharedWeightsFiles =
        pathsToFileSources(parseScalarOrSeq(sharedWeightsPathsYaml));

    // Cache Eviction
    if (hasValue(runtimeOptYaml["cacheEvictionMode"])
        || hasValue(runtimeOptYaml["localSnapKvAttnSinkSize"])) {
        // TODO: Deprecated
        LOG(WARN) << "The use of 'cacheEvictionMode' and 'localSnapKvAttnSinkSize' cache eviction "
                     "config is deprecated. Please update your cache eviction config to the latest "
                     "format.";
        auto& cacheEvictOpt = runtimeOptions.cacheEvictionOptions;
        cacheEvictOpt.mode = mtk::getCacheEvictionModeFromName(
            tryParse<std::string>(runtimeOptYaml["cacheEvictionMode"]).c_str());
        cacheEvictOpt.snapkvAttnSinkSize =
            tryParse<size_t>(runtimeOptYaml["localSnapKvAttnSinkSize"]);
    }
    if (hasValue(cacheEvictOptYaml)) {
        auto& cacheEvictOpt = runtimeOptions.cacheEvictionOptions;
        cacheEvictOpt.mode = mtk::getCacheEvictionModeFromName(
            tryParse<std::string>(cacheEvictOptYaml["mode"]).c_str());
        cacheEvictOpt.evictPromptCacheOnly =
            tryParse<bool>(cacheEvictOptYaml["evictPromptCacheOnly"]);
        cacheEvictOpt.useIngraphAttnReduce =
            tryParse<bool>(cacheEvictOptYaml["useIngraphAttnReduce"]);
        // Default set useSinkRotEmb to true if using LocalSnapKV, unless set to false by user.
        cacheEvictOpt.useSinkRotEmb =
            tryParse<bool>(cacheEvictOptYaml["useSinkRotEmb"],
                           cacheEvictOpt.mode == mtk::CacheEvictionMode::LocalSnapKV);
        cacheEvictOpt.snapkvAttnSinkSize =
            tryParse<size_t>(cacheEvictOptYaml["snapkvAttnSinkSize"]);
        cacheEvictOpt.snapkvWindowSize =
            tryParse<size_t>(cacheEvictOptYaml["snapkvWindowSize"], 32);
        cacheEvictOpt.snapkvKernelSize = tryParse<size_t>(cacheEvictOptYaml["snapkvKernelSize"], 7);
        cacheEvictOpt.snapkvMaxCapacity =
            tryParse<size_t>(cacheEvictOptYaml["snapkvMaxCapacity"], 2048);
        cacheEvictOpt.snapkvPooling =
            tryParse<std::string>(cacheEvictOptYaml["snapkvPooling"], "MaxPool");
    }
}

void YamlConfigParser::parseLlmModelPaths(LlmModelOptions& modelOptions,
                                          LlmRuntimeOptions& runtimeOptions) const {
    const auto& runtimeOptYaml = getRuntimeOptionsYaml();

    // Basic prompt & gen model config. For backward compatibility.
    const auto& dlaPromptPathsYaml = runtimeOptYaml["dlaPromptPaths"];
    const auto& dlaGenPathsYaml = runtimeOptYaml["dlaGenPaths"];
    const size_t numPromptDla = dlaPromptPathsYaml ? dlaPromptPathsYaml.size() : 0;
    const size_t numGenDla = dlaGenPathsYaml ? dlaGenPathsYaml.size() : 0;

    // Advanced model config
    const auto& dlaPathsYaml = runtimeOptYaml["dlaPaths"];
    for (const auto& kv : dlaPathsYaml) {
        const auto& modelConfig = kv.first.as<std::string>();
        const auto& dlaPaths = parseScalarOrSeq(kv.second);
        if (dlaPaths.empty())
            continue;
        runtimeOptions.dlaFiles.emplace(modelConfig, pathsToFileSources(dlaPaths));
    }

    // Basic prompt & gen model config. For backward compatibility.
    auto getModelConfig = [&](const size_t tokenSize) {
        std::ostringstream ss;
        ss << tokenSize << "t" << modelOptions.cacheSize << "c";
        return ss.str();
    };
    if (numPromptDla) {
        const auto& dlaPromptPaths = parseScalarOrSeq(dlaPromptPathsYaml);
        runtimeOptions.dlaFiles.emplace(
            getModelConfig(modelOptions.promptTokenBatchSize), pathsToFileSources(dlaPromptPaths));
    }
    if (numGenDla) {
        const auto& dlaGenPaths = parseScalarOrSeq(dlaGenPathsYaml);
        runtimeOptions.dlaFiles.emplace(
            getModelConfig(modelOptions.genTokenBatchSize), pathsToFileSources(dlaGenPaths));
    }

    auto parseTokenSize = [](const std::string& modelConfig) -> size_t {
        size_t tokenSize = 0;
        std::smatch match;
        if (std::regex_search(modelConfig, match, std::regex("([0-9]+)[tT]")))
            tokenSize = std::stoi(match[0].str());
        else
            LOG(FATAL) << "Unable to parse token size from model config: '" << modelConfig << "'";
        return tokenSize;
    };

    // Store prompt and gen token sizes based on finalized model configs
    const auto [minTokenSize, maxTokenSize] = [&]() -> std::pair<size_t, size_t> {
        CHECK_GE(runtimeOptions.dlaFiles.size(), 1);
        size_t minTokenSize = std::numeric_limits<size_t>::max();
        size_t maxTokenSize = 0;
        for (const auto& [modelConfig, _] : runtimeOptions.dlaFiles) {
            const auto tokenSize = parseTokenSize(modelConfig);
            minTokenSize = std::min(minTokenSize, tokenSize);
            maxTokenSize = std::max(maxTokenSize, tokenSize);
        }
        DCHECK_LE(minTokenSize, maxTokenSize);
        return {minTokenSize, maxTokenSize};
    }();

    modelOptions.promptTokenBatchSize = maxTokenSize;
    modelOptions.genTokenBatchSize = minTokenSize;
}

void LlavaYamlConfigParser::parseRuntimeOptions(LlmRuntimeOptions& runtimeOptions) const {
    YamlConfigParser::parseRuntimeOptions(runtimeOptions);

    auto& llavaRuntimeOptions = static_cast<LlavaRuntimeOptions&>(runtimeOptions);

    const auto& runtimeOptYaml = getRuntimeOptionsYaml();

    const auto& clipPathYaml = runtimeOptYaml["clipPath"];
    const auto& clipPatchEmbYaml = runtimeOptYaml["clipPatchEmb"];

    // Validation
    if (!clipPathYaml) {
        LOG(FATAL) << "Invalid yaml config file: 'clipPath' is not defined in the config.";
    }

    llavaRuntimeOptions.clipFile = clipPathYaml.as<std::string>();

    llavaRuntimeOptions.patchEmbFile = tryParse<std::string>(clipPatchEmbYaml);

    // Parse optional clipPreprocess config (backward-compatible: absent = LLaVA defaults)
    const auto& preprocYaml = runtimeOptYaml["clipPreprocess"];
    if (preprocYaml) {
        auto& pp = llavaRuntimeOptions.clipPreprocess;
        if (preprocYaml["mode"]) pp.mode = preprocYaml["mode"].as<std::string>();
        if (preprocYaml["resizeMode"]) pp.resizeMode = preprocYaml["resizeMode"].as<std::string>();
        if (preprocYaml["imageWidth"]) pp.imageWidth = preprocYaml["imageWidth"].as<int>();
        if (preprocYaml["imageHeight"]) pp.imageHeight = preprocYaml["imageHeight"].as<int>();
        if (preprocYaml["patchSize"]) pp.patchSize = preprocYaml["patchSize"].as<int>();
        if (preprocYaml["temporalPatchSize"]) pp.temporalPatchSize = preprocYaml["temporalPatchSize"].as<int>();
        if (preprocYaml["mergeSize"]) pp.mergeSize = preprocYaml["mergeSize"].as<int>();
        if (preprocYaml["mean"]) {
            auto m = preprocYaml["mean"];
            for (int j = 0; j < 3 && j < (int)m.size(); j++) pp.mean[j] = m[j].as<float>();
        }
        if (preprocYaml["std"]) {
            auto s = preprocYaml["std"];
            for (int j = 0; j < 3 && j < (int)s.size(); j++) pp.std[j] = s[j].as<float>();
        }
    }

    // Phase 3.5 (deepstack): optional yaml flag enabling Qwen3-VL multi-output encoder.
    // Default false → unchanged behavior for non-deepstack models.
    const auto& clipUseDeepstackYaml = runtimeOptYaml["clipUseDeepstack"];
    if (clipUseDeepstackYaml) {
        llavaRuntimeOptions.clipUseDeepstack = clipUseDeepstackYaml.as<bool>();
        LOG(INFO) << "[deepstack] clipUseDeepstack=" << llavaRuntimeOptions.clipUseDeepstack;
    }
}

} // namespace utils