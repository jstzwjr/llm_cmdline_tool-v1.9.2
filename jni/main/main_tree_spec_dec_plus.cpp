#include "common/dump.h"
#include "common/logging.h"
#include "common/timer.h"
#include "fast_drafter/fast_drafter.h"
#include "mtk_llm.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/tokenizer_factory.h"
#include "utils/NPUWareUtilsLib.h"
#include "utils/config_parser.h"
#include "utils/utils.h"

#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using TokenType = mtk::Tokenizer::TokenType;
using TokenizerUPtr = std::unique_ptr<mtk::Tokenizer>;
using mtk::LogitsKind;
using mtk::TokenizerFactory;
using AuxiliaryDrafter = mtk::AuxiliaryDrafter;

LlmModelOptions llmModelOpt;
LlmModelOptions draftModelOpt;
LlmRuntimeOptions llmRuntimeOpt;
LlmRuntimeOptions draftRuntimeOpt;

enum class SpecDecInferType : int {
    UnionMethodV1
};

enum class SpecDecVariant : int {
    Spd,
    SpdPlus
};

struct StaticTreeContext {
    std::vector<size_t> positions;
    std::vector<int> parentId;
    std::vector<std::vector<int>> mask;
    std::vector<std::vector<int>> retrieveIndices;
};

struct SpecDecContext {
    void* targetRuntime;
    void* draftRuntime;
    size_t inferenceStep;
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution;
    const size_t draftLength;
    const size_t nDraftNodes;
    const std::vector<size_t> allTopK;
    const size_t maxResponse;
    const SpecDecInferType inferType;
    const TokenizerUPtr tokenizer;
    const float targetSamplingTemperature;
    const float draftSamplingTemperature;
    SpecDecVariant spdVariant;
    AuxiliaryDrafter auxiliaryDrafter;
    const size_t draftLengthAuxDrafter;
    const size_t maxTolerance;
    const size_t minTolerance;
};

static const size_t randomSeed = 20240402;

TokenizerUPtr prepare_tokenizer(const LlmRuntimeOptions& runtimeOpt) {
    auto tokenizerInstance =
        TokenizerFactory().create(runtimeOpt.tokenizerPath, runtimeOpt.tokenizerRegex);
    const auto& specialTokens = runtimeOpt.specialTokens;
    if (specialTokens.addBos)
        tokenizerInstance->enableBosToken(specialTokens.bosId);
    return tokenizerInstance;
}

bool isStopToken(const TokenType token) {
    const auto& stopTokenSet = llmRuntimeOpt.specialTokens.stopToken;
    return stopTokenSet.find(token) != stopTokenSet.end();
};

std::tuple<std::string, std::vector<TokenType>>
get_prompt_and_tokens(const std::string& inputString, const TokenizerUPtr& tokenizer,
                      const bool parsePromptTokens) {
    // Parse or tokenize input
    const auto inputTokens =
        parsePromptTokens ? utils::parseTokenString(inputString) : tokenizer->tokenize(inputString);

    const auto& inputPrompt = parsePromptTokens ? tokenizer->detokenize(inputTokens) : inputString;
    return {inputPrompt, inputTokens};
}

void llm_init_spec_dec(void** targetRuntime, void** draftRuntime, const std::string& yamlConfigPath,
                       const std::string& yamlConfigPathDraft) {
    Timer timer;
    timer.start();
    LOG(INFO) << "Begin target model init...";

    utils::YamlConfigParser configParser(yamlConfigPath);
    configParser.parse(llmModelOpt, llmRuntimeOpt);

    bool status = mtk_llm_init(targetRuntime, llmModelOpt, llmRuntimeOpt);
    if (!status) {
        LOG(FATAL) << "LLM init failed";
    }

    LOG(INFO) << "Begin draft model init...";
    utils::YamlConfigParser draftConfigParser(yamlConfigPathDraft);
    draftConfigParser.parse(draftModelOpt, draftRuntimeOpt);

    status = mtk_llm_init(draftRuntime, draftModelOpt, draftRuntimeOpt);
    if (!status) {
        LOG(FATAL) << "LLM init failed";
    }

    double elapsed = timer.reset();
    LOG(INFO) << "Done model init. (Time taken: " << elapsed << "s)";
}

void llm_swap_model(void* llmRuntime, const size_t tokenSize = 1) {
    Timer timer;
    timer.start();
    LOG(INFO) << "Hot swapping to " << tokenSize << "t model...";
    mtk_llm_swap_model(llmRuntime, tokenSize);
    double elapsed = timer.reset();
    LOG(INFO) << "Done model hot swapping. (Time taken: " << elapsed << "s)";
}

TokenType llm_digest_prompt(SpecDecContext& ctx, const bool isTarget,
                            const std::vector<TokenType>& inputTokens, const size_t modelTokenSize,
                            double& promptTokPerSec) {
    const auto logitsType = llmModelOpt.modelOutputType;
    void* lastLogits;
    const auto inpBeginIt = inputTokens.begin();
    const auto inputTokenCount = inputTokens.size();
    size_t inputTokenIndex = 0;

    void* llmRuntime = isTarget ? ctx.targetRuntime : ctx.draftRuntime;

    auto getNewTokens = [&]() {
        // Calculate prompt tokens size for current step
        const size_t numInputTokenLeft = inputTokenCount - inputTokenIndex;
        const size_t remainder = numInputTokenLeft % modelTokenSize;
        // Construct subset prompt tokens
        const auto numNewTok = remainder ? remainder : modelTokenSize;
        const auto tokIdxStart = inputTokenIndex;       // inclusive
        const auto tokIdxEnd = tokIdxStart + numNewTok; // exclusive
        const auto newTokens = std::vector(inpBeginIt + tokIdxStart, inpBeginIt + tokIdxEnd);
        LOG(DEBUG) << "Feeding model with prompt tokens [" << tokIdxStart << " - " << tokIdxEnd
                   << "] (numToken=" << numNewTok << "): " << newTokens;
        return newTokens;
    };

    Timer promptTimer;
    promptTimer.start();
    mtk_llm_reset_tree_attn(llmRuntime);
    while (inputTokenIndex < inputTokenCount) {
        SET_DUMP_INDEX(ctx.inferenceStep++);
        LOG(DEBUG) << "Token position: " << inputTokenIndex << ": " << inputTokens[inputTokenIndex];

        const auto curInputTokens = getNewTokens();
        const auto numNewTok = curInputTokens.size();
        DUMP(INPUTS).fromVector("input_tokens", curInputTokens);
        DUMP(INPUTS).fromString("input_string", ctx.tokenizer->detokenize(curInputTokens));

        auto isLastPromptStep = [&] { return inputTokenIndex + numNewTok >= inputTokenCount; };

        // Only the last prompt step needs logits
        const auto logitsKind = isLastPromptStep() ? LogitsKind::LAST : LogitsKind::NONE;
        lastLogits = mtk_llm_inference_once(llmRuntime, curInputTokens, logitsKind);

        inputTokenIndex += numNewTok;
    }
    double promptTimeTaken = promptTimer.reset();

    // Ideal prompt size is a multiple of prompt token batch size
    const size_t idealPromptSize =
        std::ceil(float(inputTokenCount) / modelTokenSize) * modelTokenSize;
    DCHECK_EQ(idealPromptSize % modelTokenSize, 0);
    promptTokPerSec = idealPromptSize / promptTimeTaken;

    LOG(INFO) << "Done analyzing prompt in " << promptTimeTaken << "s" << " (" << promptTokPerSec
              << " tok/s)";
    // Prompt mode ended, take the output and feed as input
    // Argmax to generate the token first
    const auto outputToken =
        utils::argmaxFrom16bitLogits(logitsType, lastLogits, ctx.tokenizer->vocabSize());
    return outputToken;
}

StaticTreeContext gen_static_tree_setup(std::vector<size_t> allTopK, const size_t depth) {
    // Depth here doesn't include the root node
    StaticTreeContext tctx = {
        .positions = {0}, .parentId = {-1}, .mask = {}, .retrieveIndices = {}};
    int bias = 0;
    std::vector<int> parentPostionsThisLevel;

    for (size_t t = 0; t < depth; t++) {
        const auto topK = allTopK[t];
        // Conduct position ids
        tctx.positions.resize(tctx.positions.size() + topK, t + 1);
        // Conduct parent ID
        parentPostionsThisLevel.resize(topK, bias);
        tctx.parentId.insert(
            tctx.parentId.end(), parentPostionsThisLevel.begin(), parentPostionsThisLevel.end());
        parentPostionsThisLevel.clear();
        bias += ((t == 0) ? 1 : allTopK[t - 1]);
    }
    LOG(DEBUG) << "[Spec-Dec][Tree setup] positions: " << tctx.positions;
    LOG(DEBUG) << "[Spec-Dec][Tree setup] parentId: " << tctx.parentId;

    auto [mask, retrieveIndices] = utils::conductTreeMask(tctx.parentId);
    tctx.mask = mask;
    tctx.retrieveIndices = retrieveIndices;
    return tctx;
}

std::tuple<std::vector<TokenType>, std::vector<float>> llm_spec_dec_gen_static_tree_candidate(
    SpecDecContext& ctx, const bool singleCandidate, const size_t vocabSize,
    const float draftOutputQuantScale, const float draftSamplingTemperature, TokenType& outputToken,
    double& meanDraftElapsed, double& meanDraftRunOnceElapsed, double& meanGenDraftTokensElapsed) {
    const auto draftLogitsType = draftModelOpt.modelOutputType;
    const auto draftGenTokenSize = draftModelOpt.genTokenBatchSize;
    std::vector<TokenType> draftTokens;

    std::vector<TokenType> eachRunTokens = {outputToken};
    std::vector<std::vector<int>> nextStepMask = {{1}};
    std::vector<size_t> eachStepPositions = {0};
    size_t tokenSizeOffset = 0;

    std::vector<float> draftProbs;

    Timer timerDraft, timerDraftRunOnce, timerGenDraftTokens;
    double allDraftTimePerStep = 0, allDraftRunOnce = 0, allGenDraftTokens = 0;

    auto conductIdentityMask = [&](const size_t topK) -> std::vector<std::vector<int>> {
        std::vector<std::vector<int>> identityMask(topK, std::vector<int>(topK, 0));
        for (size_t i = 0; i < topK; i++) {
            identityMask[i][i] = 1;
        }
        return identityMask;
    };

    for (size_t t = 0; t < ctx.draftLength; t++) {
        const auto topK = ctx.allTopK[t];
        // Set next step token offset for positions
        tokenSizeOffset += ((t < 2) ? 0 : (ctx.allTopK[t - 2] - 1));

        // Set attention mask and postions
        mtk_llm_reset_tree_attn(ctx.draftRuntime);
        mtk_llm_set_tree_attn(
            ctx.draftRuntime, nextStepMask, eachStepPositions, draftGenTokenSize, tokenSizeOffset);

        timerDraft.start();
        timerDraftRunOnce.start();
        void* draftLastLogits =
            mtk_llm_inference_once(ctx.draftRuntime, eachRunTokens, LogitsKind::FULL);
        auto draftRunOnceElapsed = timerDraftRunOnce.reset() * 1000;

        eachRunTokens.clear();

        timerGenDraftTokens.start();
        if (ctx.inferType == SpecDecInferType::UnionMethodV1) {
            if (singleCandidate) {
                // Signle candidate
                const auto [updatedToken, tokenProb] = utils::randomSampleFrom16bitLogits(
                    draftLogitsType, draftLastLogits, vocabSize, draftOutputQuantScale,
                    draftSamplingTemperature);
                outputToken = updatedToken;
                draftProbs.push_back(tokenProb);
                draftTokens.push_back(outputToken);
                eachRunTokens.push_back(outputToken);
            } else {
                // Tree candidate
                auto [topkTokens, topkProbs] = utils::getTopkArgmaxAndLogitsV2(
                    draftLogitsType, draftLastLogits, vocabSize, topK, draftOutputQuantScale,
                    draftSamplingTemperature);
                outputToken = topkTokens[0];
                draftProbs.insert(draftProbs.end(), topkProbs.begin(), topkProbs.end());
                draftTokens.insert(draftTokens.end(), topkTokens.begin(), topkTokens.end());
                eachRunTokens.insert(eachRunTokens.begin(), topkTokens.begin(), topkTokens.end());
            }
        }
        auto genDraftTokensElapsed = timerGenDraftTokens.reset() * 1000;

        // Set next step mask
        const auto previousStepMask = nextStepMask;
        nextStepMask.resize(topK);
        for (size_t rowId = 0; rowId < topK; rowId++) {
            nextStepMask[rowId] = previousStepMask[0];
        }
        auto identityMask = conductIdentityMask(topK);
        nextStepMask = utils::horizontalConcat(nextStepMask, identityMask);

        // Set next step positions
        eachStepPositions.resize(topK, 0);

        DCHECK_EQ(eachRunTokens.size(), topK) << "len of eachRunTokens should be the same as topk";
        auto draftElapsed = timerDraft.reset() * 1000;
        LOG(DEBUG) << "[Spec-Dec][Draft]: Generate the " << t
                   << "-th draft token. Time elapsed: " << draftElapsed;

        allDraftTimePerStep += draftElapsed;
        allDraftRunOnce += draftRunOnceElapsed;
        allGenDraftTokens += genDraftTokensElapsed;
    }
    meanDraftElapsed = allDraftTimePerStep / ctx.draftLength;
    meanDraftRunOnceElapsed = allDraftRunOnce / ctx.draftLength;
    meanGenDraftTokensElapsed = allGenDraftTokens / ctx.draftLength;

    LOG(DEBUG) << "[Spec-Dec][Draft]: Complete the generation. Tokens:" << draftTokens;

    return {draftTokens, draftProbs};
}

std::tuple<std::vector<size_t>, TokenType>
llm_spec_dec_judgment(StaticTreeContext& treeInfos, SpecDecContext& ctx,
                      const size_t curDraftLength, const std::string whichDrafting,
                      const std::vector<TokenType>& draftTokens,
                      const std::vector<float>& draftProbs, void* targetLogits, size_t& acceptNum,
                      double& genTargetTokensElapsed, double& judgmentElapsed) {
    Timer timerJudgment, timerGenTargetTokens;
    timerJudgment.start();
    // Judgment
    const auto targetLogitsType = llmModelOpt.modelOutputType;
    const float targetOutputQuantScale = llmModelOpt.modelOutputQuantScale;
    const float targetSamplingTemperature = ctx.targetSamplingTemperature;
    const auto vocabSize = ctx.tokenizer->vocabSize();
    const auto logitsBuffer = reinterpret_cast<const char*>(targetLogits);
    const size_t logitsSize = mtk_llm_get_per_token_logits_size(ctx.targetRuntime);

    const size_t nDraftTokens = ctx.nDraftNodes;
    TokenType bestPromiseToken;
    size_t bestGeneratedLength = 0;
    size_t bestCandidateInd = 0;

    // TRICK3: Use this DS to avoid the repeated operations in CPU.
    auto curTopK = (whichDrafting == "spd") ? ctx.allTopK : std::vector<size_t>(curDraftLength, 1);
    std::vector<int> parentLevelKeys = {-1};
    std::vector<size_t> LeftToRightIndex = {0};
    std::vector<size_t> accumulateNodes = {1};
    size_t parentLevel = 0;
    for (size_t topkId = 0; topkId < curTopK.size(); topkId++) {
        auto topK = curTopK[topkId];
        // Parent level keys matching
        parentLevelKeys.resize(parentLevelKeys.size() + topK, parentLevel);
        parentLevel++;
        // Which index on a node's level
        std::vector<int> thisLevelIndices(topK);
        std::iota(thisLevelIndices.begin(), thisLevelIndices.end(), 0);
        LeftToRightIndex.insert(
            LeftToRightIndex.end(), thisLevelIndices.begin(), thisLevelIndices.end());
        // accumulateNodes
        accumulateNodes.push_back(accumulateNodes.back() + topK);
    }
    DCHECK_EQ(parentLevelKeys.size(), nDraftTokens);
    DCHECK_EQ(LeftToRightIndex.size(), nDraftTokens);

    std::vector<std::tuple<bool, std::vector<TokenType>, std::vector<float>>> targetTokenStatus(
        curDraftLength, std::make_tuple(false, std::vector<TokenType>(), std::vector<float>()));
    // The end of varaible call for trick3

    TokenType promiseToken;

    DCHECK_EQ(draftTokens.size(), nDraftTokens) << "draftTokens len should equal to nDraftTokens.";
    DCHECK_EQ(draftProbs.size(), nDraftTokens) << "draftProbs len should equal to nDraftTokens.";

    // TRICK3: randomSample take argmax token and p(draft Token indices of a specific level)
    // E.g: topk=4, 2, 1. Draft token = 01234567, target tokens = abcdefgh
    // We did randomSample operation to get tokenid of index "a" (need it in bit-true verification)
    // and the probabilities of "1234" on the distribution of "a" (need this as executing p(x)/q(x)
    // , where x = 1, 2, 3, or 4) at the same time.
    // This function checks if current candidate and its corresponding node already been
    // calculated before. The status is recorded in targetTokenStatus.
    auto genTargetInfosAndUpdateStatus =
        [&](const size_t ind, const std::vector<int>& candidate) -> std::pair<TokenType, float> {
        TokenType targetTokenId;
        float targetTokenProb;
        auto curNodeParentLevel = parentLevelKeys[candidate[ind]];
        auto idFromLeftToRight = LeftToRightIndex[candidate[ind]];

        if (std::get<0>(targetTokenStatus[curNodeParentLevel])) {
            // This level already did randomSample. Just take the corresponding token
            // id and prob
            targetTokenId = std::get<1>(targetTokenStatus[curNodeParentLevel])[0];
            targetTokenProb = std::get<2>(targetTokenStatus[curNodeParentLevel])[idFromLeftToRight];
        } else {
            // This level hasn't performed random sampling yet.
            const auto curTargetLogits = logitsBuffer + candidate[ind - 1] * logitsSize;
            auto start = accumulateNodes[ind - 1];
            auto end = accumulateNodes[ind];
            timerGenTargetTokens.start();
            std::vector<TokenType> draftTokenIndices(
                draftTokens.begin() + start, draftTokens.begin() + end);
            const auto [answerToken, tokenProbs] = utils::randomSampleFrom16bitLogits(
                targetLogitsType, curTargetLogits, vocabSize, targetOutputQuantScale,
                targetSamplingTemperature, draftTokenIndices);
            genTargetTokensElapsed += timerGenTargetTokens.elapsed() * 1000;
            targetTokenId = answerToken;
            targetTokenProb = tokenProbs[idFromLeftToRight];

            std::get<0>(targetTokenStatus[curNodeParentLevel]) = true;
            std::get<1>(targetTokenStatus[curNodeParentLevel]) = {targetTokenId};
            std::get<2>(targetTokenStatus[curNodeParentLevel]) = tokenProbs;
        }
        return {targetTokenId, targetTokenProb};
    };

    for (size_t idx = 0; idx < treeInfos.retrieveIndices.size(); idx++) {
        const auto candidate = treeInfos.retrieveIndices[idx];
        size_t generatedLength = 1;
        const auto correspondingDraftTokens = utils::gather(draftTokens, candidate);
        const auto correspondingDraftProbs = utils::gather(draftProbs, candidate);
        std::vector<TokenType> correspondingTargetTokens;
        std::vector<float> correspondingTargetProbs;
        for (size_t ind = 1; ind < correspondingDraftTokens.size(); ind++) {
            /*
            topk = 4, 2, 1
            draftTokens  = {d0, d1, d2, d3, d4, d5, d6, d7}
            targetTokens = {t1, t2, t3, t4, t5, t6, t7, t8}
            Assume candidate = {d0, d1, d6}, then corresponding target tokens: {t1, t2, t7}
            Judgment: (d1==t1 || p(d1) / q(d1)), (d6 == t2 || p(d6) / q(d6))...
            */
            // Gen target tokens and probs
            TokenType updatedToken;
            float tokenProb;
            if (ctx.inferType == SpecDecInferType::UnionMethodV1) {
                auto targetInfos = genTargetInfosAndUpdateStatus(ind, candidate);
                updatedToken = targetInfos.first;
                tokenProb = targetInfos.second;
            }
            correspondingTargetTokens.push_back(updatedToken);
            correspondingTargetProbs.push_back(tokenProb);
            // Judgment
            const bool acceptDraftToken =
                ((correspondingDraftTokens[ind] == correspondingTargetTokens[ind - 1])
                 || (ctx.distribution(ctx.generator)
                     < correspondingTargetProbs[ind - 1] / correspondingDraftProbs[ind]));

            if (acceptDraftToken && !isStopToken(correspondingDraftTokens[ind])) {
                generatedLength += 1;
                if (generatedLength == candidate.size()) {
                    // All accept
                    const auto curTargetLogits = logitsBuffer + candidate.back() * logitsSize;
                    timerGenTargetTokens.start();
                    promiseToken = utils::randomSampleFrom16bitLogits(
                                       targetLogitsType, curTargetLogits, vocabSize,
                                       targetOutputQuantScale, targetSamplingTemperature)
                                       .first;
                    genTargetTokensElapsed += timerGenTargetTokens.elapsed() * 1000;
                }
            } else {
                // Not all accept
                promiseToken = correspondingTargetTokens[ind - 1];
                break; // If reject one token, stop inner loop.
            }
        }
        // Choose the best candidate and corresponding infos.
        if (generatedLength > bestGeneratedLength) {
            bestGeneratedLength = generatedLength;
            bestCandidateInd = idx;
            bestPromiseToken = promiseToken;
            const auto nextIdx = std::min(idx + 1, treeInfos.retrieveIndices.size());
            const auto nextCandidateSize = treeInfos.retrieveIndices[nextIdx].size();
            DCHECK_GE(candidate.size(), nextCandidateSize);
            if (generatedLength >= nextCandidateSize) {
                // TRICK2: If current generated length > nextCandidateSize, we don't need to run
                // the remaining candidates.
                // ASSUMPTION: This condition holds only when candidate[i - 1].size() always
                // greater or equal to candidate[i]
                break;
            }
        }
    }

    auto bestCandidate = treeInfos.retrieveIndices[bestCandidateInd];
    std::vector<size_t> generatedIndices;
    for (size_t i = 0; i < bestGeneratedLength; i++) {
        generatedIndices.push_back(bestCandidate[i]);
    }
    acceptNum = generatedIndices.size() - 1;
    judgmentElapsed = timerJudgment.reset() * 1000;

    return {generatedIndices, bestPromiseToken};
}

std::tuple<std::vector<TokenType>, TokenType>
llm_spec_dec_per_step(SpecDecContext& ctx, StaticTreeContext& treeInfos, const bool singleCandidate,
                      const TokenType inputToken, size_t& acceptNum, double& meanDraftElapsed,
                      double& meanDraftRunOnceElapsed, double& meanGenDraftTokensElapsed,
                      double& verificationElapsed, double& parallelDecodingElapsed,
                      double& genTargetTokensElapsed, double& judgmentElapsed,
                      double& adjustCacheElapsed, std::vector<std::string>& allStepDrafting) {
    Timer timerVerification, timerParallelDecoding, timerGenTargetTokens, timerJudgment;
    Timer timerAdjustCache;

    const float draftOutputQuantScale = draftModelOpt.modelOutputQuantScale;
    const size_t genTokenSize = llmModelOpt.genTokenBatchSize;
    const float draftSamplingTemperature = ctx.draftSamplingTemperature;
    const auto vocabSize = ctx.tokenizer->vocabSize();
    const auto allTopK = ctx.allTopK;
    TokenType outputToken = inputToken;

    std::vector<TokenType> tokensToVerify;
    void* targetLogits;
    std::vector<TokenType> draftTokens;
    std::vector<float> draftProbs;

    std::string whichDrafting = "none";
    size_t curDraftLength = ctx.draftLength;

    LOG(DEBUG) << "[Spec-Dec]: The newest token (confirmedNewToken) is: " << inputToken;

    // SPD+ Drafting
    if (ctx.spdVariant == SpecDecVariant::SpdPlus) {
        whichDrafting = "aux";
        Timer timerDraft;
        timerDraft.start();
        draftTokens = ctx.auxiliaryDrafter.getDraftTokens(
            ctx.maxTolerance, ctx.minTolerance, ctx.draftLengthAuxDrafter);
        draftProbs.resize(ctx.draftLengthAuxDrafter, 0.0f);
        meanDraftElapsed = timerDraft.reset() * 1000;
        meanDraftRunOnceElapsed = 0;
        meanGenDraftTokensElapsed = 0;
    }

    if (draftTokens.empty()) {
        whichDrafting = "spd";
        auto draftInfos = llm_spec_dec_gen_static_tree_candidate(
            ctx, singleCandidate, vocabSize, draftOutputQuantScale, draftSamplingTemperature,
            outputToken, meanDraftElapsed, meanDraftRunOnceElapsed, meanGenDraftTokensElapsed);
        draftTokens = std::get<0>(draftInfos);
        draftProbs = std::get<1>(draftInfos);
    }

    // Fix current draft length
    if (whichDrafting == "spd") {
        curDraftLength = ctx.draftLength;
    } else if (whichDrafting == "aux") {
        curDraftLength = ctx.draftLengthAuxDrafter;
    } else {
        LOG(FATAL) << "Drafting way should be spd or aux!";
    }

    // Target model verifies the draft tokens.
    tokensToVerify.push_back(inputToken);
    tokensToVerify.insert(tokensToVerify.end(), draftTokens.begin(), draftTokens.end());
    tokensToVerify.resize(genTokenSize);
    LOG(DEBUG) << "[Spec-Dec][Target] Input Tokens: " << tokensToVerify;

    // Set tree attn for the target model
    if ((whichDrafting == "spd") && (!singleCandidate)) {
        // Tree candidate
        treeInfos = gen_static_tree_setup(ctx.allTopK, curDraftLength);
        treeInfos.positions.resize(genTokenSize, 0);
        mtk_llm_set_tree_attn(ctx.targetRuntime, treeInfos.mask, treeInfos.positions, genTokenSize);

    } else if (((whichDrafting == "spd") && (singleCandidate)) || (whichDrafting == "aux")) {
        // Single cnadidate
        mtk_llm_reset_tree_attn(ctx.targetRuntime);
        std::vector<int> singleRetrieveIndices(curDraftLength + 1);
        std::iota(singleRetrieveIndices.begin(), singleRetrieveIndices.end(), 0);
        treeInfos.retrieveIndices.push_back(singleRetrieveIndices);
    }

    timerVerification.start();
    timerParallelDecoding.start();
    targetLogits = mtk_llm_inference_once(ctx.targetRuntime, tokensToVerify, LogitsKind::FULL);

    parallelDecodingElapsed = timerParallelDecoding.elapsed() * 1000;
    LOG(DEBUG) << "[Spec-Dec][Target]: Latency of Target(" << genTokenSize
               << "-T): " << parallelDecodingElapsed << " ms.";
    acceptNum = 0;

    draftTokens.insert(draftTokens.begin(), inputToken);
    draftProbs.insert(draftProbs.begin(), 1.0);

    auto [acceptedIndices, promiseToken] = llm_spec_dec_judgment(
        treeInfos, ctx, curDraftLength, whichDrafting, draftTokens, draftProbs, targetLogits,
        acceptNum, genTargetTokensElapsed, judgmentElapsed);
    // Root node of next step = this step promise token
    outputToken = promiseToken;
    LOG(DEBUG) << "[Spec-Dec] acceptNum: " << acceptNum;

    // Adjust acceptTokens
    auto acceptedTokens = utils::gather(draftTokens, acceptedIndices);
    acceptedTokens.erase(acceptedTokens.begin());
    // acceptedTokens has no root node, and has no promise token.

    verificationElapsed = timerVerification.reset() * 1000;
    LOG(DEBUG) << "[Spec-Dec][Verification]: Latency for verification: " << verificationElapsed
               << " ms.";
    LOG(DEBUG) << "[Spec-Dec][Judgment]: Accepted tokens: " << acceptNum;
    allStepDrafting.push_back(whichDrafting);

    // Stop when any of accepted tokens is a stop token (default to EoS if not set)
    if (isStopToken(outputToken)) {
        std::cout << "</eos>";
        return {acceptedTokens, outputToken};
    }

    timerAdjustCache.start();
    mtk_llm_reset_tree_attn(ctx.draftRuntime);

    // Manipulate the cache.
    // Adjust cache: Draft model
    if (whichDrafting == "spd") {
        auto draftRetainIndices = acceptedIndices;
        if (acceptNum == curDraftLength) {
            // All accept
            draftRetainIndices.pop_back();
            mtk_llm_tree_rollback(
                ctx.draftRuntime, draftRetainIndices, ctx.nDraftNodes - ctx.allTopK.back());
            mtk_llm_inference_once(ctx.draftRuntime, {acceptedTokens.back()}, LogitsKind::NONE);
        } else if (acceptNum < ctx.draftLength) {
            // Not all accept
            mtk_llm_tree_rollback(
                ctx.draftRuntime, draftRetainIndices, ctx.nDraftNodes - ctx.allTopK.back());
        }
    } else if (whichDrafting == "aux") {
        // Draft model cache: use >1t gen mode to fill draft model cache
        std::vector<TokenType> lackCacheTokens = {inputToken};
        lackCacheTokens.insert(lackCacheTokens.end(), acceptedTokens.begin(), acceptedTokens.end());
        const size_t draftInputTokenSize = draftModelOpt.genTokenBatchSize;
        for (size_t i = 0; i < lackCacheTokens.size(); i += draftInputTokenSize) {
            std::vector<TokenType> tokenChunk;
            for (size_t j = i; j < i + draftInputTokenSize && j < lackCacheTokens.size(); j++) {
                tokenChunk.push_back(lackCacheTokens[j]);
            }
            mtk_llm_inference_once(ctx.draftRuntime, tokenChunk, LogitsKind::NONE);
        }
    }
    // Target model cache: rollback
    auto targetRetainIndices = acceptedIndices;
    mtk_llm_tree_rollback(ctx.targetRuntime, targetRetainIndices);

    adjustCacheElapsed = timerAdjustCache.reset() * 1000;

    LOG(DEBUG) << "[Spec-Dec][Rollback]: Latency overhead: " << adjustCacheElapsed << " ms.";
    return {acceptedTokens, outputToken};
}

void llm_gen_response(SpecDecContext& ctx, const TokenType firstInputToken, double& genTokPerSec) {
    const size_t maxTokenLength = llmModelOpt.maxTokenLength;
    auto curTokenIndex = mtk_llm_get_token_index(ctx.targetRuntime);
    const size_t& sequenceLength = curTokenIndex;

    double elapsed = 0, genTotalTime = 0;
    genTokPerSec = 0;
    size_t genTokCount = 0, specDecCount = 0;
    TokenType totalAcceptNum = 0, allAcceptNum = 0;
    double totalDraftTime = 0, totalDraftRunOnceTime = 0, totalGenDraftTokensTime = 0;
    double totalTargetTime = 0, totalParallelDecodingTime = 0, totalTargetTokensTime = 0,
           totalJudgmentTime = 0;
    double totalAdjustCacheTime = 0;

    std::string fullResponse;
    utils::UTF8CharResolver utf8Resolver;
    TokenType outputToken = firstInputToken;
    std::vector<size_t> generatedNums;
    std::vector<std::string> allStepDrafting;
    size_t spdNums = 0, spdCounts = 0;
    size_t auxNums = 0, auxCounts = 0;

    const bool singleCandidate =
        std::all_of(ctx.allTopK.begin(), ctx.allTopK.end(), [](int i) { return i == 1; });

    Timer timer;
    timer.start();
    while (genTokCount < ctx.maxResponse && sequenceLength < maxTokenLength) {
        StaticTreeContext treeInfos;
        if (ctx.inferType == SpecDecInferType::UnionMethodV1) {
            SET_DUMP_INDEX(ctx.inferenceStep++);
            // Save and print outputToken
            const std::string tokStr = ctx.tokenizer->detokenize(outputToken);
            const bool isTokStrResolved = utf8Resolver.addBytes(tokStr);
            if (isTokStrResolved) {
                const std::string response = utf8Resolver.getResolvedStr();
                std::cout << response << std::flush;
                fullResponse += response;
                DUMP(RESPONSE).fromValue("sampled_token", outputToken);
                DUMP(RESPONSE).fromString("sampled_text", tokStr);
                DUMP(RESPONSE).fromString("full_response", fullResponse);
            }
            // update auxDrafter if needed
            if (ctx.spdVariant == SpecDecVariant::SpdPlus) {
                ctx.auxiliaryDrafter.update({outputToken});
            }

            size_t acceptNum;
            double meanDraftElapsed, meanDraftRunOnceElapsed, meanGenDraftTokensElapsed;
            double verificationElapsed, parallelDecodingElapsed, genTargetTokensElapsed = 0;
            double judgmentElapsed, adjustCacheElapsed;
            auto [acceptedTokens, lastAcceptToken] = llm_spec_dec_per_step(
                ctx, treeInfos, singleCandidate, outputToken, acceptNum, meanDraftElapsed,
                meanDraftRunOnceElapsed, meanGenDraftTokensElapsed, verificationElapsed,
                parallelDecodingElapsed, genTargetTokensElapsed, judgmentElapsed,
                adjustCacheElapsed, allStepDrafting);
            outputToken = lastAcceptToken;

            // Save and print all accepted tokens in this decoding step.
            for (const auto acceptedToken : acceptedTokens) {
                const std::string tokStr = ctx.tokenizer->detokenize(acceptedToken);
                const bool isTokStrResolved = utf8Resolver.addBytes(tokStr);
                if (isTokStrResolved) {
                    const std::string response = utf8Resolver.getResolvedStr();
                    std::cout << response << std::flush;
                    fullResponse += response;
                    DUMP(RESPONSE).fromValue("sampled_token", acceptedToken);
                    DUMP(RESPONSE).fromString("sampled_text", tokStr);
                    DUMP(RESPONSE).fromString("full_response", fullResponse);
                }
            }
            if (ctx.spdVariant == SpecDecVariant::SpdPlus) {
                ctx.auxiliaryDrafter.update(acceptedTokens);
            }
            specDecCount++;
            genTokCount += (acceptNum + 1);
            curTokenIndex += (acceptNum + 1);
            generatedNums.push_back(acceptNum + 1);

            totalAcceptNum += acceptNum;
            if (allStepDrafting.back() == "aux") {
                auxNums += acceptNum;
                auxCounts++;
                if (acceptNum == ctx.draftLengthAuxDrafter) {
                    allAcceptNum++;
                }
            } else if (allStepDrafting.back() == "spd") {
                spdNums += acceptNum;
                spdCounts++;
                if (acceptNum == ctx.draftLength) {
                    allAcceptNum++;
                }
            } else {
                LOG(FATAL) << "Incorrect drafting way";
            }

            totalDraftTime += meanDraftElapsed;
            totalDraftRunOnceTime += meanDraftRunOnceElapsed;
            totalGenDraftTokensTime += meanGenDraftTokensElapsed;
            totalTargetTime += verificationElapsed;
            totalParallelDecodingTime += parallelDecodingElapsed;
            totalTargetTokensTime += genTargetTokensElapsed;
            totalJudgmentTime += judgmentElapsed;
            totalAdjustCacheTime += adjustCacheElapsed;

            elapsed = timer.reset();
            genTotalTime += elapsed;
            LOG(DEBUG) << "Single loop time taken: " << elapsed * 1000 << " ms";

            // Stop when output is a stop token (default to EoS if not set)
            if (isStopToken(outputToken)) {
                std::cout << "</eos>";
                break;
            }
        }
    }
    std::cout << "</end>" << std::endl;
    genTokPerSec = double(genTokCount) / genTotalTime;
    float spdProportion = static_cast<float>(spdCounts) / specDecCount;
    float auxProportion = static_cast<float>(auxCounts) / specDecCount;

    if (ctx.inferType == SpecDecInferType::UnionMethodV1) {
        std::cout << "\n[Full Response]\n" << fullResponse << std::endl;
        std::cout << "\n[Info]" << std::endl;
        std::cout << "      Avg. Accpeted num:  " << double(totalAcceptNum) / (specDecCount)
                  << std::endl;
        std::cout << "      Avg. Accpeted rate: "
                  << double(totalAcceptNum)
                         / (auxCounts * ctx.draftLengthAuxDrafter + spdCounts * ctx.draftLength)
                  << std::endl;
        std::cout << "      All-accept Rate:    " << double(allAcceptNum) / specDecCount
                  << std::endl;
        std::cout << "\n[Latency]" << std::endl;
        std::cout << "      Drafting latency:                  " << totalDraftTime / specDecCount
                  << " ms" << std::endl;
        std::cout << "          Run model once latency:        "
                  << totalDraftRunOnceTime / specDecCount << " ms" << std::endl;
        std::cout << "          Gen draft tokens latency:      "
                  << totalGenDraftTokensTime / specDecCount << " ms" << std::endl;
        std::cout << "      Verification latency:              " << totalTargetTime / specDecCount
                  << " ms" << std::endl;
        std::cout << "          Parallel decoding latency:     "
                  << totalParallelDecodingTime / specDecCount << " ms" << std::endl;
        std::cout << "          Judgment latency:              " << totalJudgmentTime / specDecCount
                  << " ms" << std::endl;
        std::cout << "              Gen target tokens latency: "
                  << totalTargetTokensTime / specDecCount << " ms" << std::endl;
        std::cout << "      Adjust cache latency:              "
                  << totalAdjustCacheTime / specDecCount << " ms" << std::endl;
        std::cout << "\n[Stats]" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "      Aux drafter count: " << auxCounts << " (" << auxProportion * 100 << "%)"
                  << std::endl;
        std::cout << "      SpD drafter count: " << spdCounts << " (" << spdProportion * 100 << "%)"
                  << std::endl;
        if (auxCounts == 0) {
            std::cout << "      Aux - Avg. Accpeted num: 0" << std::endl;
        } else {
            std::cout << "      Aux - Avg. Accpeted num:  " << double(auxNums) / auxCounts
                      << std::endl;
        }
        if (spdCounts == 0) {
            std::cout << "      SpD - Avg. Accpeted num: 0" << std::endl;
        } else {
            std::cout << "      SpD - Avg. Accpeted num:  " << double(spdNums) / spdCounts
                      << std::endl;
        }
        std::cout << std::defaultfloat << std::setprecision(6);
    }
    LOG(DEBUG) << "[Spec-Dec][Acceptance] generatedNums: " << generatedNums;
    LOG(DEBUG) << "[Spec-Dec][SpD ratio] allStepDrafting: " << allStepDrafting;
}

std::tuple<double, double> llm_inference_spec_dec(
    void* targetRuntime, void* draftRuntime, const SpecDecInferType inferType,
    const size_t draftLength, const std::vector<size_t>& allTopK, const std::string& inputString,
    const size_t maxResponse = 50, const bool parsePromptTokens = false,
    const float upperBound = 1.0, const float targetSamplingTemperature = 0.0,
    const float draftSamplingTemperature = 0.0, const size_t draftLengthAuxDrafter = 0,
    const size_t maxTolerance = 0, const size_t minTolerance = 0) {
    SpecDecContext ctx = {
        .targetRuntime = targetRuntime,
        .draftRuntime = draftRuntime,
        .generator = std::default_random_engine(randomSeed),
        .distribution = std::uniform_real_distribution<float>(0.0, upperBound),
        .draftLength = draftLength,
        .nDraftNodes = static_cast<size_t>(std::accumulate(allTopK.begin(), allTopK.end(), 1)),
        .allTopK = allTopK,
        .maxResponse = maxResponse,
        .inferType = inferType,
        .tokenizer = prepare_tokenizer(llmRuntimeOpt),
        .targetSamplingTemperature = targetSamplingTemperature,
        .draftSamplingTemperature = draftSamplingTemperature,
        .auxiliaryDrafter = mtk::AuxiliaryDrafter(),
        .draftLengthAuxDrafter = draftLengthAuxDrafter,
        .maxTolerance = maxTolerance,
        .minTolerance = minTolerance,
    };
    // Sanity check for tree attn
    if (ctx.nDraftNodes > llmModelOpt.genTokenBatchSize) {
        LOG(FATAL)
            << "nDraftNodes (includes root node) should be less or equal to genTokenBatchSize"
            << "(" << "nDraftNodes: " << ctx.nDraftNodes
            << ", genTokenBatchSize: " << llmModelOpt.genTokenBatchSize << ")";
    }
    const auto maxTopk = *std::max_element(allTopK.begin(), allTopK.end());
    if (draftModelOpt.genTokenBatchSize < maxTopk) {
        LOG(FATAL)
            << "genTokenBatchSize of gen mode for the draft model should be larger than maxTopk";
    }
    // Sanity check for spd+
    if ((ctx.maxTolerance == ctx.minTolerance) && (ctx.minTolerance == ctx.draftLengthAuxDrafter)
        && (ctx.draftLengthAuxDrafter == 0)) {
        ctx.spdVariant = SpecDecVariant::Spd;
        LOG(INFO) << "vanialla SPD is using now!";
    } else {
        ctx.spdVariant = SpecDecVariant::SpdPlus;
        LOG(INFO) << "SPD+ is using now!";
    }

    ctx.auxiliaryDrafter.reset();

    // Prepare tokenizers for both models
    const auto draftTokenizer = prepare_tokenizer(draftRuntimeOpt);

    const auto& tokenizer = ctx.tokenizer;

    CHECK_EQ(tokenizer->vocabSize(), draftTokenizer->vocabSize())
        << "Different vocab size for the target and the draft model.";

    // Convert string to tokens
    auto [draftInputPrompt, draftInputTokens] =
        get_prompt_and_tokens(inputString, draftTokenizer, parsePromptTokens);

    auto [inputPrompt, inputTokens] =
        get_prompt_and_tokens(inputString, tokenizer, parsePromptTokens);

    CHECK_EQ(inputPrompt, draftInputPrompt)
        << "target model and the draft model may be using different tokenizers!";
    CHECK_EQ(inputTokens, draftInputTokens)
        << "target model and the draft model may be using different tokenizers!";
    DUMP(PROMPT).fromVector("prompt_tokens", inputTokens);
    DUMP(PROMPT).fromString("prompt_text", inputPrompt);

    std::cout << "\n[Prompt]\n" << inputPrompt << '\n' << std::endl;

    // Draft Model: input prompt caching
    ctx.inferenceStep = 0;
    size_t draftPromptTokenSize = draftModelOpt.promptTokenBatchSize;
    double draftPromptTokPerSec;
    llm_digest_prompt(
        ctx, /*isTarget*/ false, draftInputTokens, draftPromptTokenSize, draftPromptTokPerSec);

    // Draft Model: Swap to gen mode if model is still in prompt mode.
    const size_t draftGenTokenSize = draftModelOpt.genTokenBatchSize;
    if (draftPromptTokenSize != draftGenTokenSize) {
        llm_swap_model(ctx.draftRuntime, draftGenTokenSize);
    }

    // Target Model: input prompt caching
    ctx.inferenceStep = 0;
    size_t promptTokenSize = llmModelOpt.promptTokenBatchSize;
    double promptTokPerSec;
    const TokenType outputToken =
        llm_digest_prompt(ctx, /*isTarget*/ true, inputTokens, promptTokenSize, promptTokPerSec);
    // Target Model: Swap to gen mode if model is still in prompt mode.
    const size_t genTokenSize = llmModelOpt.genTokenBatchSize;
    CHECK_GT(genTokenSize, ctx.draftLength)
        << "genTokenSize in target model config should be larger than draftlen!";
    if (promptTokenSize != genTokenSize) {
        llm_swap_model(ctx.targetRuntime, genTokenSize);
    }

    const double totalPromptTokPerSec =
        1.0 / ((1.0 / promptTokPerSec) + (1.0 / draftPromptTokPerSec));

    // Update AuxDrafter if Spd+
    if (ctx.spdVariant == SpecDecVariant::SpdPlus) {
        ctx.auxiliaryDrafter.update(inputTokens);
    }

    // Generation process
    std::cout << "\nResponse [Max Length = " << ctx.maxResponse << "]:" << std::endl;
    double genTokPerSec;
    llm_gen_response(ctx, outputToken, genTokPerSec);

    ctx.auxiliaryDrafter.reset();

    std::cout << "\n[Speed]" << std::endl;
    std::cout << "      Prompt Mode: " << totalPromptTokPerSec << " tok/s" << std::endl;
    std::cout << "  Generative Mode: " << genTokPerSec << " tok/s" << std::endl;
    return {totalPromptTokPerSec, genTokPerSec};
}

void llm_reset(void* llmRuntime) {
    mtk_llm_reset(llmRuntime);
}

void llm_release(void* llmRuntime) {
    mtk_llm_release(llmRuntime);
}

int main(int argc, char* argv[]) {
    ScopePerformancer scopePerformancer; // Enable PowerHAL

    std::string yamlConfigPath = "config.yaml";
    std::string yamlConfigPathDraft = "";
    SpecDecInferType inferType = SpecDecInferType::UnionMethodV1;
    size_t maxResponse = 200;
    bool parsePromptTokens = false; // Read prompt as a string of tokens
    bool onePromptPerLine = false;  // Treat each line in prompt text as a single prompt. Will
                                    // replace literal "\n" with new line char '\n'.
    std::string preformatterName = "";
    size_t draftLen = 0;
    size_t draftLenAuxDrafter = 0;
    size_t maxTolerance = 0;
    size_t minTolerance = 0;
    float upperBound = 1.0;               // Threshold ~ U[0, upperBound]
    std::vector<std::string> promptPaths; // Paths containing the prompt text
    std::vector<std::string> prompts;
    // std::string prompt = "Once upon a time,";
    const std::string defaultPrompt = "Tell me about alpacas";
    float targetSamplingTemperature = 0.0;
    float draftSamplingTemperature = 0.0;
    std::vector<size_t> allTopK;

    using utils::matchArgument;

    // Process command line.
    //  -m or --max to set the max response.
    //  -p or --prompt to set the input prompt.
    //  -i or --input-file to set the path to the text containing the input prompt.
    //  --read-tokens to read the input prompt as a string of tokens.
    //  --one-prompt-per-line to treat each line in prompt file as one prompt. The literal "\n" is
    //  treated as new line.
    //  --infer-type to set inference type of Spec-Dec.
    //  '-d or --draft' and '-r or --draft-len' should be initialized if --infer-type is 0.
    for (int i = 1; i < argc; i++) {
        std::string curArg(argv[i]);
        if (matchArgument(curArg, "--max", "-m")) {
            ENSURE_NEXT_ARG_EXISTS(i)
            maxResponse = std::atoi(argv[++i]);
        } else if (matchArgument(curArg, "--prompt", "-p")) {
            ENSURE_NEXT_ARG_EXISTS(i)
            prompts.emplace_back(argv[++i]);
        } else if (matchArgument(curArg, "--input-file", "-i")) {
            ENSURE_NEXT_ARG_EXISTS(i)
            promptPaths.emplace_back(argv[++i]);
        } else if (matchArgument(curArg, "--infer-type")) {
            ENSURE_NEXT_ARG_EXISTS(i)
            inferType = static_cast<SpecDecInferType>(std::atoi(argv[++i]));
        } else if (matchArgument(curArg, "--draft", "-d")) {
            ENSURE_NEXT_ARG_EXISTS(i)
            yamlConfigPathDraft = argv[++i];
            LOG(INFO) << "Using yaml config file for draft model: " << yamlConfigPathDraft;
        } else if (matchArgument(curArg, "--draft-len", "-r")) {
            ENSURE_NEXT_ARG_EXISTS(i)
            draftLen = std::atoi(argv[++i]);
            LOG(INFO) << "Draft length: " << draftLen;
        } else if (matchArgument(curArg, "--draft-len-aux", "-r")) {
            ENSURE_NEXT_ARG_EXISTS(i)
            draftLenAuxDrafter = std::atoi(argv[++i]);
            LOG(INFO) << "Auxiliary drafter length: " << draftLenAuxDrafter;
        } else if (matchArgument(curArg, "--max-tol", "-r")) {
            ENSURE_NEXT_ARG_EXISTS(i)
            maxTolerance = std::atoi(argv[++i]);
            LOG(INFO) << "maxTolerance: " << maxTolerance;
        } else if (matchArgument(curArg, "--min-tol", "-r")) {
            ENSURE_NEXT_ARG_EXISTS(i)
            minTolerance = std::atoi(argv[++i]);
            LOG(INFO) << "minTolerance: " << minTolerance;
        } else if (fs::path(curArg).extension() == ".yaml") {
            LOG(INFO) << "Using yaml config file: " << curArg;
            yamlConfigPath = curArg;
        } else if (matchArgument(curArg, "--read-tokens", "-t")) {
            parsePromptTokens = true;
        } else if (matchArgument(curArg, "--one-prompt-per-line")) {
            onePromptPerLine = true;
        } else if (matchArgument(curArg, "--preformatter")) {
            ENSURE_NEXT_ARG_EXISTS(i)
            preformatterName = argv[++i];
        } else if (matchArgument(curArg, "--upper-bound")) {
            ENSURE_NEXT_ARG_EXISTS(i)
            upperBound = std::atof(argv[++i]);
            LOG(INFO) << "Using upper bound: " << upperBound;
        } else if (matchArgument(curArg, "--target-temperature")) {
            ENSURE_NEXT_ARG_EXISTS(i)
            targetSamplingTemperature = std::atof(argv[++i]);
            LOG(INFO) << "Using temperature for target model: " << targetSamplingTemperature;
            LOG(WARN) << "Remember to specify the modelOutputQuantScale in the target yaml file,"
                      << " or the results maybe incorrect in some cases! (e.g. 4w16a model)";
        } else if (matchArgument(curArg, "--draft-temperature")) {
            ENSURE_NEXT_ARG_EXISTS(i)
            draftSamplingTemperature = std::atof(argv[++i]);
            LOG(INFO) << "Using temperature for draft model: " << draftSamplingTemperature;
            LOG(WARN) << "Remember to specify the modelOutputQuantScale in the draft yaml file,"
                      << " or the results maybe incorrect in some cases! (e.g. 4w16a model)";
        } else if (matchArgument(curArg, "--tree-topk")) {
            ENSURE_NEXT_ARG_EXISTS(i)
            const auto tmpTopks = utils::parseTokenString(argv[++i]);
            for (auto topk : tmpTopks)
                allTopK.push_back(static_cast<size_t>(topk));
        } else {
            LOG(INFO) << "Unrecognized argument: " << curArg;
        }
    }
    if (allTopK.empty() || (allTopK.size() == 1 && allTopK[0] == 1))
        allTopK.resize(draftLen, 1);

    if (allTopK.size() != draftLen) {
        LOG(FATAL) << "Number of elements in allTopK should equal to draft length. (allTopK size: "
                   << allTopK.size() << ", draft len: " << draftLen << ")";
    }

    prompts = utils::readPromptFiles(promptPaths, onePromptPerLine);

    if (prompts.empty())
        prompts.push_back(defaultPrompt); // Use the default example.

    double allPromptTokPerSec = 0, allGenTokPerSec = 0;
    const size_t numPrompt = prompts.size();
    void* llmRuntime = nullptr;
    void* draftLlmRuntime = nullptr;
    switch (inferType) {
        case SpecDecInferType::UnionMethodV1:
            llm_init_spec_dec(&llmRuntime, &draftLlmRuntime, yamlConfigPath, yamlConfigPathDraft);
            break;
        default:
            LOG(FATAL) << "Wrong INFER_METHOD initialized the bat file, or this main file doesn't "
                          "suppoort this method";
    }

    for (size_t i = 0; i < numPrompt; i++) {
        std::cout << "============ Processing the " << i << "-th input. ============" << std::endl;
        std::string prompt = prompts[i];
        DUMP(PROMPT).fromString("text", prompt);
        if (!parsePromptTokens && !preformatterName.empty()) {
            if (utils::addPreformatter(preformatterName, prompt)) {
                LOG(INFO) << "Preformatted prompt with '" << preformatterName << "'";
                DUMP(PROMPT).fromString("text_preformatted", prompt);
            } else {
                LOG(ERROR) << "Invalid preformatter: '" << preformatterName << "'";
            }
        }
        switch (inferType) {
            case SpecDecInferType::UnionMethodV1: {
                LOG(INFO) << "Sanity check...";
                CHECK_GT(draftLen, 0) << "Need to specify draft_len in bat file.";
                CHECK(!yamlConfigPathDraft.empty())
                    << "Need to specify draft model (--draft) in bat file.";

                const auto [promptTokPerSec, genTokPerSec] = llm_inference_spec_dec(
                    llmRuntime, draftLlmRuntime, inferType, draftLen, allTopK, prompt, maxResponse,
                    parsePromptTokens, upperBound, targetSamplingTemperature,
                    draftSamplingTemperature, draftLenAuxDrafter, maxTolerance, minTolerance);
                allPromptTokPerSec += promptTokPerSec;
                allGenTokPerSec += genTokPerSec;
                llm_reset(llmRuntime);
                llm_reset(draftLlmRuntime);
                llm_swap_model(llmRuntime, llmModelOpt.promptTokenBatchSize);
                llm_swap_model(draftLlmRuntime, draftModelOpt.promptTokenBatchSize);
                break;
            }
            default: {
                LOG(FATAL) << "Wrong INFER_METHOD initialized the bat file.";
            }
        }
    }
    llm_release(llmRuntime);
    if (inferType == SpecDecInferType::UnionMethodV1) {
        llm_release(draftLlmRuntime);
    }
    std::cout << "\n[Average Performance among the given " << numPrompt << " prompts]\n";
    std::cout << "      Prompt Mode: " << allPromptTokPerSec / numPrompt << " tok/s\n";
    std::cout << "  Generative Mode: " << allGenTokPerSec / numPrompt << " tok/s\n";
}