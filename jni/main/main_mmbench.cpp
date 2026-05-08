// main_mmbench.cpp
// 端侧 MMBench 多选题精度评测器（专用 binary，与 main_llava 同目录构建）。
//
// 输入：TSV，每行一题
//     row<TAB>image_path<TAB>prompt<TAB>letters<TAB>gt
//   row         ：整型行号（与主机端 meta.jsonl 对齐）
//   image_path  ：手机端绝对路径
//   prompt      ：完整提示文本，**真实换行已转义为字面 "\n"**（与 readPromptFiles 约定一致）
//                 不含 chat 模板，binary 内部按 --preformatter 加上模板
//   letters     ：本题候选字母集合（"AB" / "ABCD" / ...），用于限制 argmax 范围
//   gt          ：正确答案字母（"A"/"B"/"C"/"D"），可空（test split 无 GT）
//
// 输出（stdout 或 -o 指定的文件）：
//   每题一行：
//     MMBENCH_RESULT<TAB>row<TAB>pred<TAB>gt<TAB>correct<TAB>prompt_ms<TAB>num_prompt_tokens
//     correct ∈ {1, 0, -}（- 表示无 GT）
//   末尾汇总（仅有 GT 的样本计入）：
//     MMBENCH_SUMMARY<TAB>correct<TAB>total<TAB>accuracy
//
// 评测口径：仅 prefill，对 last_logits 在 {A,B,C,D}（Qwen3 token id 32/33/34/35）
//          子集上做 argmax；不做自回归生成，0 token gen。
//          每题 mtk_mllm_reset(runtime, true)，避免 KV cache 污染。
//
// 用法：
//   ./main_mmbench config.yaml -i input.tsv [-o results.tsv] \
//                  [--preformatter QwenNoInput] [--progress 50]

#include "common/logging.h"
#include "common/timer.h"
#include "mtk_llm_options.h"
#include "mtk_mllm.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/tokenizer_factory.h"
#include "utils/NPUWareUtilsLib.h"
#include "utils/config_parser.h"
#include "utils/utils.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using TokenType = mtk::Tokenizer::TokenType;
using TokenizerUPtr = std::unique_ptr<mtk::Tokenizer>;
using mtk::TokenizerFactory;

LlmModelOptions llmModelOpt;
LlavaRuntimeOptions llavaMllmRuntimeOpt;

// Qwen3 tokenizer 中 "A"/"B"/"C"/"D" 的单 token id（连续 32~35）
static constexpr TokenType kABCDTokenIds[4] = {32, 33, 34, 35};
static constexpr char kABCDLetters[4] = {'A', 'B', 'C', 'D'};

// 与 readPromptFiles 反解析对齐：把字面 "\\n" 还原成真实换行；"\\\\" 还原成 "\\"
std::string unescape_newlines(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (i + 1 < s.size() && s[i] == '\\' && s[i + 1] == 'n') {
            out.push_back('\n');
            ++i;
        } else if (i + 1 < s.size() && s[i] == '\\' && s[i + 1] == '\\') {
            out.push_back('\\');
            ++i;
        } else {
            out.push_back(s[i]);
        }
    }
    return out;
}

std::vector<std::string> split_tsv(const std::string& line, const char sep = '\t') {
    std::vector<std::string> out;
    size_t start = 0;
    while (true) {
        size_t end = line.find(sep, start);
        if (end == std::string::npos) {
            out.emplace_back(line.substr(start));
            break;
        }
        out.emplace_back(line.substr(start, end - start));
        start = end + 1;
    }
    return out;
}

TokenizerUPtr prepare_tokenizer(const LlmRuntimeOptions& runtimeOpt) {
    auto tokenizer = TokenizerFactory().create(runtimeOpt.tokenizerPath, runtimeOpt.tokenizerRegex);
    const auto& specialTokens = runtimeOpt.specialTokens;
    if (specialTokens.addBos)
        tokenizer->enableBosToken(specialTokens.bosId);
    return tokenizer;
}

// 把 prompt 中所有 <image> 替换为 kImagePlaceholderToken；其余文本走 tokenizer
std::vector<TokenType> tokenize_with_image(const std::string& promptText,
                                           const TokenizerUPtr& tokenizer) {
    static constexpr char imageTag[] = "<image>";
    static constexpr size_t imageTagLen = sizeof(imageTag) - 1;
    const auto& specialTokens = llavaMllmRuntimeOpt.specialTokens;

    std::vector<TokenType> inputTokens;
    if (specialTokens.addBos) {
        inputTokens.push_back(specialTokens.bosId);
    }

    size_t pos = 0;
    while (pos < promptText.size()) {
        size_t found = promptText.find(imageTag, pos);
        if (found == std::string::npos) {
            std::string tail = promptText.substr(pos);
            if (!tail.empty()) {
                auto toks = tokenizer->tokenize(tail);
                inputTokens.insert(inputTokens.end(), toks.begin(), toks.end());
            }
            break;
        }
        if (found > pos) {
            auto chunk = promptText.substr(pos, found - pos);
            auto toks = tokenizer->tokenize(chunk);
            inputTokens.insert(inputTokens.end(), toks.begin(), toks.end());
        }
        inputTokens.push_back(kImagePlaceholderToken);
        pos = found + imageTagLen;
    }
    return inputTokens;
}

// 在 last_logits 上对受限的 letters 子集做 argmax，返回胜出字母（'?' 表示无候选）
char pick_choice(const void* lastLogits, const mtk::LLMType logitsType,
                 const std::string& allowedLetters) {
    float bestVal = -1e30f;
    char bestLetter = '?';
    for (size_t i = 0; i < 4; ++i) {
        if (allowedLetters.find(kABCDLetters[i]) == std::string::npos)
            continue;
        const TokenType id = kABCDTokenIds[i];
        float v;
        if (logitsType == mtk::LLMType::FP16) {
            v = static_cast<float>(reinterpret_cast<const __fp16*>(lastLogits)[id]);
        } else if (logitsType == mtk::LLMType::INT16) {
            v = static_cast<float>(reinterpret_cast<const int16_t*>(lastLogits)[id]);
        } else {
            v = 0.0f;
        }
        if (v > bestVal) {
            bestVal = v;
            bestLetter = kABCDLetters[i];
        }
    }
    return bestLetter;
}

int main(int argc, char* argv[]) {
    ScopePerformancer scopePerformancer; // 启用 PowerHAL

    std::string yamlConfigPath;
    std::string inputTsvPath;
    std::string outputPath;
    // 默认 QwenNoInput（向后兼容历史 79.50% baseline）。
    // Qwen3-VL 应在调用时显式 --preformatter Qwen3VLNoInput --image-style qwen3vl
    // 启用与 HF 训练模板完全对齐的版本（详见 utils.cpp:addPreformatter_Qwen3VLNoInput）。
    std::string preformatterName = "QwenNoInput";
    // 默认空字符串。CLI 解析完后，若仍为空则按 preformatter 名自动派生
    // （详见 utils::defaultImageStyleFor）：Qwen3VLNoInput→qwen3vl，其他→bare。
    // 用户可显式 --image-style xxx 覆盖（用于消融/调试）。
    std::string imageStyle = "";
    int progressEvery = 50;

    using utils::matchArgument;
    for (int i = 1; i < argc; ++i) {
        std::string curArg(argv[i]);
        if (matchArgument(curArg, "--input", "-i")) {
            ENSURE_NEXT_ARG_EXISTS(i)
            inputTsvPath = argv[++i];
        } else if (matchArgument(curArg, "--output", "-o")) {
            ENSURE_NEXT_ARG_EXISTS(i)
            outputPath = argv[++i];
        } else if (matchArgument(curArg, "--preformatter", "-pref")) {
            ENSURE_NEXT_ARG_EXISTS(i)
            preformatterName = argv[++i];
        } else if (matchArgument(curArg, "--image-style")) {
            ENSURE_NEXT_ARG_EXISTS(i)
            imageStyle = argv[++i];
        } else if (matchArgument(curArg, "--progress")) {
            ENSURE_NEXT_ARG_EXISTS(i)
            progressEvery = std::atoi(argv[++i]);
        } else if (fs::path(curArg).extension() == ".yaml") {
            yamlConfigPath = curArg;
        } else {
            LOG(WARN) << "Unrecognized argument: " << curArg;
        }
    }

    if (yamlConfigPath.empty() || inputTsvPath.empty()) {
        std::cerr << "Usage: main_mmbench <config.yaml> -i <input.tsv> [-o <output.tsv>] "
                  << "[--preformatter QwenNoInput] [--image-style bare|qwen3vl] "
                  << "[--progress 50]" << std::endl;
        return 1;
    }

    // image-style 未显式指定则按 preformatter 名自动派生（Qwen3VLNoInput→qwen3vl，其他→bare）。
    if (imageStyle.empty()) {
        imageStyle = utils::defaultImageStyleFor(preformatterName);
        LOG(INFO) << "image-style auto-derived from preformatter '" << preformatterName
                  << "' -> '" << imageStyle << "'";
    } else {
        LOG(INFO) << "image-style explicitly set to '" << imageStyle << "'";
    }

    // 解析 yaml
    utils::LlavaYamlConfigParser configParser(yamlConfigPath);
    configParser.parse(llmModelOpt, llavaMllmRuntimeOpt);

    // 初始化 mllm（一次）
    Timer initTimer;
    initTimer.start();
    LOG(INFO) << "Begin mllm init...";
    void* mllmRuntime = nullptr;
    if (!mtk_mllm_init(&mllmRuntime, llmModelOpt, llavaMllmRuntimeOpt)) {
        LOG(FATAL) << "mtk_mllm_init failed";
        return 1;
    }
    LOG(INFO) << "Done mllm init. (Time taken: " << initTimer.reset() << "s)";

    auto tokenizer = prepare_tokenizer(llavaMllmRuntimeOpt);
    LOG(INFO) << "Vocab size: " << tokenizer->vocabSize();
    const auto logitsType = llmModelOpt.modelOutputType;

    std::ifstream fin(inputTsvPath);
    if (!fin) {
        LOG(FATAL) << "Cannot open input file: " << inputTsvPath;
        return 1;
    }
    std::ofstream fout;
    if (!outputPath.empty()) {
        fout.open(outputPath);
        if (!fout) {
            LOG(FATAL) << "Cannot open output file: " << outputPath;
            return 1;
        }
    }
    std::ostream& outStream = outputPath.empty() ? std::cout : fout;

    // 跳过表头（如果存在）
    std::string firstLine;
    if (std::getline(fin, firstLine)) {
        // 表头形如 "row\timage_path\tprompt\tletters"，其首字段非数字
        bool isHeader = !firstLine.empty() && (firstLine[0] < '0' || firstLine[0] > '9');
        if (!isHeader) {
            // 把第一行还原为待处理
            fin.clear();
            fin.seekg(0);
        }
    }

    Timer overall;
    overall.start();
    Timer perTimer;
    size_t count = 0;
    size_t scoredTotal = 0;
    size_t scoredCorrect = 0;

    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        auto fields = split_tsv(line);
        if (fields.size() < 4) {
            LOG(WARN) << "Malformed line (need 4-5 cols): " << line.substr(0, 100);
            continue;
        }

        const std::string& rowStr = fields[0];
        const std::string& imagePath = fields[1];
        std::string promptRaw = unescape_newlines(fields[2]);
        const std::string& letters = fields[3];
        const std::string gt = (fields.size() >= 5) ? fields[4] : std::string();

        // 在最前面插入 <image>，再套 chat 模板。
        // <image> 是占位符，在下游 tokenize_with_image 中被替换为 kImagePlaceholderToken，
        // 再于 prefill 时展开为 N 个 image_pad 槽。
        // 不同 VL 模型对图像的包裹方式不同（bare / qwen3vl 等），通过 imageStyle 控制：
        //   - "bare"：插入 "<image>\n" + prompt（LLaVA 类，向后兼容历史 79.50% baseline）。
        //   - "qwen3vl"：插入 "<|vision_start|><image><|vision_end|>" + prompt（无 \n，
        //                与 HF Qwen3-VL 训练 chat 模板字节级对齐：vision_end 紧贴 prompt）。
        // 故意按 style 直接拼最终格式，不走"bare→applyImageStyle 改写"的路径，避免
        // \n 残留破坏 vision_end 与 prompt 的紧贴关系。
        std::string prompt;
        if (imageStyle == "qwen3vl") {
            prompt = std::string("<|vision_start|><image><|vision_end|>") + promptRaw;
        } else {
            prompt = std::string("<image>\n") + promptRaw;
        }
        if (!preformatterName.empty()) {
            if (!utils::addPreformatter(preformatterName, prompt)) {
                LOG(ERROR) << "Invalid preformatter: " << preformatterName;
            }
        }

        auto inputTokens = tokenize_with_image(prompt, tokenizer);
        std::vector<std::string> imagePaths{imagePath};

        mtk_mllm_reset(mllmRuntime, /*resetCache=*/true);

        size_t numPromptToken = 0;
        perTimer.start();
        void* lastLogits = mtk_mllm_consume_prompt(mllmRuntime, inputTokens, imagePaths,
                                                   &numPromptToken);
        const double promptSec = perTimer.elapsed();

        const char pred = pick_choice(lastLogits, logitsType, letters);

        // 端侧打分
        const char* correctStr = "-";
        if (!gt.empty() && gt != "-") {
            ++scoredTotal;
            if (gt.size() == 1 && pred == gt[0]) {
                ++scoredCorrect;
                correctStr = "1";
            } else {
                correctStr = "0";
            }
        }

        outStream << "MMBENCH_RESULT\t" << rowStr << "\t" << pred << "\t"
                  << (gt.empty() ? "-" : gt) << "\t" << correctStr << "\t"
                  << static_cast<int>(promptSec * 1000) << "\t" << numPromptToken << "\n";
        outStream.flush();

        ++count;
        if (progressEvery > 0 && count % static_cast<size_t>(progressEvery) == 0) {
            const double el = overall.elapsed();
            const double avg = el / count;
            const double acc = scoredTotal > 0
                ? static_cast<double>(scoredCorrect) / scoredTotal : 0.0;
            std::cerr << "[" << count << "] acc=" << acc << " (" << scoredCorrect << "/"
                      << scoredTotal << ")  avg=" << avg << "s/题  elapsed=" << el << "s"
                      << std::endl;
        }
    }

    const double total = overall.elapsed();
    const double finalAcc = scoredTotal > 0
        ? static_cast<double>(scoredCorrect) / scoredTotal : 0.0;
    outStream << "MMBENCH_SUMMARY\t" << scoredCorrect << "\t" << scoredTotal << "\t"
              << finalAcc << "\n";
    outStream.flush();
    std::cerr << "Done. Processed " << count << " questions in " << total << "s ("
              << (count > 0 ? total / count : 0.0) << "s/题). "
              << "Accuracy = " << finalAcc << " (" << scoredCorrect << "/" << scoredTotal << ")"
              << std::endl;

    mtk_mllm_release(mllmRuntime);
    return 0;
}
