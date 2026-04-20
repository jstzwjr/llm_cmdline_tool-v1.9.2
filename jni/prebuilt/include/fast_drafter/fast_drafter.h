#pragma once

#include "../tokenizer/tokenizer.h"

#include <map>
#include <memory>
#include <vector>

using TokenType = mtk::Tokenizer::TokenType;

namespace mtk {

class AuxiliaryDrafter {
public:
    AuxiliaryDrafter();
    ~AuxiliaryDrafter();

    void update(const std::vector<TokenType>& tokens);
    void reset();
    std::vector<TokenType> getDraftTokens(const size_t maxTol, const size_t minTol,
                                          const size_t draftLength);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace mtk