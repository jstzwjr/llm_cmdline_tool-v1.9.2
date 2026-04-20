#pragma once

#include "common/file_source.h"
#include "mtk_llm_options.h"
#include "third_party/include/yaml-cpp/yaml.h"

#include <string>
#include <vector>

namespace utils {

class YamlConfigParser {
private:
    static constexpr char kModelOptionsKey[] = "modelOptions";
    static constexpr char kRuntimeOptionsKey[] = "runtimeOptions";

public:
    explicit YamlConfigParser(const std::string& configYamlPath)
        : kYamlConfig(YAML::LoadFile(configYamlPath)) {}

    virtual ~YamlConfigParser() {}

    void parse(LlmModelOptions& modelOptions, LlmRuntimeOptions& runtimeOptions) const;

    virtual void parseModelOptions(LlmModelOptions& modelOptions) const;

    virtual void parseRuntimeOptions(LlmRuntimeOptions& runtimeOptions) const;

protected:
    void parseLlmModelPaths(LlmModelOptions& modelOptions, LlmRuntimeOptions& runtimeOptions) const;

    const YAML::Node getRuntimeOptionsYaml() const { return kYamlConfig[kRuntimeOptionsKey]; }

    const YAML::Node getModelOptionsYaml() const { return kYamlConfig[kModelOptionsKey]; }

protected:
    //===---===//
    //  Utils
    //===---===//
    static bool hasValue(const YAML::Node& node) { return node.IsDefined() && !node.IsNull(); }

    static std::vector<std::string> parseScalarOrSeq(const YAML::Node& node) {
        if (!hasValue(node))
            return {};
        if (node.IsSequence())
            return node.as<std::vector<std::string>>();
        return {node.as<std::string>()};
    }

    static std::vector<FileSource> pathsToFileSources(const std::vector<std::string>& paths) {
        std::vector<FileSource> fileSources;
        fileSources.reserve(paths.size());
        for (const auto& path : paths) {
            fileSources.emplace_back(path);
        }
        return fileSources;
    }

    template <typename Type>
    static Type tryParse(const YAML::Node& node, const Type& defaultValue = Type{}) {
        return hasValue(node) ? node.as<Type>() : defaultValue;
    }

private:
    YAML::Node kYamlConfig;
};

class LlavaYamlConfigParser : public YamlConfigParser {
public:
    // Inherit parent class constructor
    using YamlConfigParser::YamlConfigParser;

    virtual void parseRuntimeOptions(LlmRuntimeOptions& runtimeOptions) const override;
};

} // namespace utils