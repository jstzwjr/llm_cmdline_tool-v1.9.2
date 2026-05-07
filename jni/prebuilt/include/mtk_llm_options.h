#pragma once

#include "common/file_source.h"
#include "mtk_llm_types.h"
#include "tokenizer/tokenizer.h"

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// clang-format off

struct LlmModelOptions {
    // Sizes
    size_t genModelBatchSize    = 1; // Currently batch inference is only supported by gen mode
    size_t promptTokenBatchSize = 1;
    size_t genTokenBatchSize    = 1;
    size_t cacheSize            = 512;
    size_t hiddenSize           = 4096;
    size_t numHead              = 32;
    size_t numLayer             = 32;
    size_t headDim              = 0;
    size_t maxTokenLength       = 2048;
    size_t numMedusaHeads       = 0;

    // Rotary Embedding
    float rotEmbBase            = 10000.0;
    float ntkScale              = 1.0;
    size_t rotEmbNumInputs      = 2;

    // Inputs
    bool splitMask              = false;

    // Used for cache eviction. Either 0, 1, or numLayer.
    // If set as numLayer, then it will be divided by the number of chunks.
    size_t numAttnOutputs       = 0;

    // Types
    mtk::LLMType modelInputType  = mtk::LLMType::INT16;
    mtk::LLMType modelOutputType = mtk::LLMType::INT16;
    mtk::LLMType cacheType       = mtk::LLMType::INT16;
    mtk::LLMType maskType        = mtk::LLMType::INT16;
    mtk::LLMType rotEmbType      = mtk::LLMType::INT16;

    // Quantization
    float embOutputQuantScale   = 0; // default 0 means not used
    float modelOutputQuantScale = 1;
};

using LoraKey = std::string;
using ModelConfig = std::string; // Example "128t1024c"
using ChunkFiles = std::vector<FileSource>;
using TokenSet = std::unordered_set<mtk::Tokenizer::TokenType>;

struct LlmRuntimeOptions {
    struct SpecialTokens {
        mtk::Tokenizer::TokenType bosId = 1; // Beginning of Sentence Token Id
        mtk::Tokenizer::TokenType eosId = 2; // End of Sentence Token Id
        bool addBos = false; // Whether BoS token will be prepended during tokenization
        TokenSet stopToken;  // Inference stops once the model generates a stop token
    } specialTokens;

    std::string tokenizerRegex;             // Optional
    std::vector<std::string> tokenizerPath; // Either a directory or file path(s)

    FileSource tokenEmbFile;
    std::unordered_map<ModelConfig, ChunkFiles> dlaFiles;
    FileSource dlaLmHeadFile;
    FileSource dlaMedusaHeadsFile;
    int startTokenIndex = 0;
    ChunkFiles cacheFiles; // Each file is a concatenation of all caches in a chunk.
    ChunkFiles sharedWeightsFiles;
    bool useLmHeadSharedWeights = false;

    LoraKey initWithLoraKey;
    size_t loraInputCount = 0; // Per DLA chunk
    std::unordered_map<LoraKey, ChunkFiles> loraWeightsFiles;

    // Optional Path to NeuroPilot Backend Runtime Library. Currently only used by USDK.
    std::string npLibPath;
    // Cache Eviction
    struct CacheEvitionOptions {
        // Mode selection
        mtk::CacheEvictionMode mode = mtk::CacheEvictionMode::None;

        bool evictPromptCacheOnly = false;
        bool useIngraphAttnReduce = false;
        bool useSinkRotEmb        = false;

        // LocalSnapKV
        size_t snapkvAttnSinkSize   = 0;
        // GlobalSnapKV
        size_t snapkvWindowSize     = 32;
        size_t snapkvKernelSize     = 7;
        size_t snapkvMaxCapacity    = 2048;
        std::string snapkvPooling   = "MaxPool";
    } cacheEvictionOptions;
};

struct ClipPreprocessOptions {
    std::string mode = "default";  // "default" (LLaVA) or "qwen_vl"
    std::string resizeMode = "stretch";  // "stretch" (direct resize) or "padding" (aspect-ratio + black padding)
    int imageWidth = 336;
    int imageHeight = 336;
    int patchSize = 14;
    int temporalPatchSize = 2;
    int mergeSize = 2;
    float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};  // OpenAI CLIP RGB
    float std[3] = {0.26862954f, 0.26130258f, 0.27577711f};   // OpenAI CLIP RGB
};

struct LlavaRuntimeOptions : public LlmRuntimeOptions {
    FileSource clipFile;
    FileSource patchEmbFile;
    ClipPreprocessOptions clipPreprocess;

    // Phase 3.1 (deepstack): when true, treat the encoder DLA as the Qwen3-VL
    // multi-output variant (1 main + 6 extras; last 3 are projector deepstack embeds).
    // Default false → existing single-output encoder behavior unchanged.
    bool clipUseDeepstack = false;
};

// clang-format on