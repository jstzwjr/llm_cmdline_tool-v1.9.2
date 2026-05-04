#pragma once

#include "common/file_source.h"
#include "mtk_llm_options.h"
#include "mtk_llm_types.h"
#include "tokenizer/tokenizer.h"

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

template <typename T>
using Batched = std::vector<T>;

struct SharedWeightsHandle;

void mtk_llm_shared_weights_create_handle(SharedWeightsHandle** sharedWeightsHandle,
                                          const LlmRuntimeOptions& runtimeOptions);

void mtk_llm_shared_weights_preload(SharedWeightsHandle** sharedWeightsHandle,
                                    const LlmRuntimeOptions& runtimeOptions);

void mtk_llm_shared_weights_preload_union(const std::vector<SharedWeightsHandle*>& preloadHandles,
                                          const std::vector<SharedWeightsHandle*>& refHandles = {});

void mtk_llm_shared_weights_unload(SharedWeightsHandle* sharedWeightsHandle, const size_t index);

void mtk_llm_shared_weights_release_handle(SharedWeightsHandle* sharedWeightsHandle);

bool mtk_llm_init(void** runtime, const LlmModelOptions& modelOptions,
                  const LlmRuntimeOptions& runtimeOptions,
                  const SharedWeightsHandle* preloadedSharedWeights = nullptr);

void mtk_llm_release(void* runtime);

void mtk_llm_set_medusa_tree_attn(void* runtime, const std::vector<std::vector<int>>& mask,
                                  const std::vector<size_t>& positions);

void mtk_llm_set_tree_attn(void* runtime, const std::vector<std::vector<int>>& mask,
                           const std::vector<size_t>& positions, const size_t modelTokenSize = 0,
                           const size_t tokenOffset = 0);

void mtk_llm_reset_tree_attn(void* runtime);

void mtk_llm_use_prompt_as_batch_gen(void* runtime);

void* mtk_llm_consume_emb(void* runtime, const char* embBuffer, const size_t embSize,
                          const mtk::LogitsKind outputKind = mtk::LogitsKind::LAST);

void* mtk_llm_inference_once(void* runtime,
                             const std::vector<mtk::Tokenizer::TokenType>& inputTokens,
                             const mtk::LogitsKind outputKind = mtk::LogitsKind::LAST);

Batched<void*>
mtk_llm_inference_batch(void* runtime,
                        const Batched<std::vector<mtk::Tokenizer::TokenType>>& batchInputTokens,
                        const mtk::LogitsKind outputKind = mtk::LogitsKind::LAST);

std::tuple<void*, void*>
mtk_llm_inference_once_return_hidden(void* runtime,
                                     const std::vector<mtk::Tokenizer::TokenType>& inputTokens,
                                     const mtk::LogitsKind outputKind = mtk::LogitsKind::LAST);

void* neuron_medusa_heads_inference_once(void* runtime, void* hiddenState);

void mtk_llm_swap_model(void* runtime, const size_t tokenSize = 1, const size_t cacheSize = 0);

size_t mtk_llm_advance_cache_size(void* runtime);

void mtk_llm_apply_lora(void* runtime, const LoraKey& loraKey);

void mtk_llm_apply_lora_from_buffer(void* runtime,
                                    const std::vector<const char*>& loraWeightBuffers,
                                    const std::vector<size_t>& sizes);

void mtk_llm_remove_lora(void* runtime);

void mtk_llm_get_caches(void* runtime, std::vector<std::vector<char*>>& caches,
                        size_t& byteSizePerCache);

std::vector<size_t> mtk_llm_get_nopad_caches_nbytes(void* runtime, const size_t numTokens);

size_t mtk_llm_save_nopad_caches(void* runtime, const size_t numTokens,
                                 const std::vector<char*>& dstBuffers);

void mtk_llm_restore_caches(void* runtime, const size_t numCachedTokens,
                            const std::vector<const char*>& cacheBuffers,
                            const std::vector<size_t>& cacheBufferSizes);

void mtk_llm_reset(void* runtime, const bool resetCache = true);

size_t mtk_llm_get_per_token_logits_size(void* runtime);

size_t mtk_llm_get_per_token_hidden_states_size(void* runtime);

size_t mtk_llm_get_token_index(void* runtime);

void mtk_llm_rollback(void* runtime, const size_t rollbackCount);

void mtk_llm_medusa_rollback(void* runtime, const std::vector<size_t>& acceptedIndices);

void mtk_llm_tree_rollback(void* runtime, const std::vector<size_t>& acceptedIndices,
                           const size_t relativeBeginOffset = 0);

bool mtk_llm_is_enabled_cache_eviction(void* runtime);

void mtk_llm_hint_num_prompt_tokens(void* runtime, const size_t count);
