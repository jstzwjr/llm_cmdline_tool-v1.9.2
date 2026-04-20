#pragma once

#include "common/thread_pool.h"
#include "llm_helper/include/utils.h"

#include <algorithm>
#include <array>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace mtk::llm_helper {

struct KVCacheInfo {
    char* buffer = nullptr;
    std::array<uint32_t, 4> shape;
    size_t typeSize = 0;

    size_t strideSize() const { return shape[3] * typeSize; }
    size_t numRows() const { return shape[0] * shape[1]; }
};

enum class CacheShrinkMode {
    // Shrink the cache towards the left.
    // The start address of a cache buffer will remain.
    // Example:
    //   Before evict: [ ][ ][A][x][C]
    //    After evict: [ ][ ][A][C][ ]
    Left,

    // Shrink the cache towards the right.
    // The end address of a cache buffer will remain.
    // Example:
    //   Before evict: [ ][ ][A][x][C]
    //    After evict: [ ][ ][ ][A][C]
    Right,

    // Shrink the cache that minimize memory movement.
    // Both start and end address may change.
    // Example:
    //   Before evict: [ ][ ][A][B][x][D]
    //    After evict: [ ][ ][A][B][D][ ]
    Adaptive
};

struct CacheEvictionStatus {
    size_t numEvictedTokens = 0;
    size_t overallCacheShift = 0; // Shift size (in tokens) towards the right
};

class CacheEvictionAgent {
public:
    explicit CacheEvictionAgent(const std::vector<KVCacheInfo>& kvCaches);

    virtual ~CacheEvictionAgent();

    // Update LLM Context
    void updateContext(const size_t tokenSize, const size_t cacheSize, const size_t numCachedTokens,
                       const size_t leftPadSize = 0, const size_t rightPadSize = 0);

    // Update KV Cache buffers
    void updateCacheBuffers(const std::vector<char*>& cacheBuffers);

    // Update Attention Weights
    void updateAttnWeights(const std::vector<std::string_view>& attnWeights,
                           const std::array<uint32_t, 4>& shape);

    // Run the cache eviction algorithm. Returns the eviction status.
    virtual CacheEvictionStatus run() = 0;

    // Reset the cache eviction algorithm state
    virtual void reset() = 0;

    // Check if the cache eviction agent requires the KV cache to be updated before triggering.
    virtual bool requiresUpdatedKVCache() const = 0;

    // Evict KV Cache from relative positions of the cached tokens.
    // For example, given cache size of 5 and 3 cached tokens, evict relative token position 1:
    //   Cached token pos  :          0   1   2
    //   Cache token slots : [ ] [ ] [C] [C] [C]
    //   Relative evict pos:              1
    void evictCacheRel(const std::vector<size_t>& relativePositions,
                       const CacheShrinkMode shrinkMode);

    // Evict KV Cache from absolute positions of the entire cache.
    // For example, given cache size of 5 and 3 cached tokens, evict absolute token position 3:
    //   Full cache pos    :  0   1   2   3   4
    //   Cache token slots : [ ] [ ] [C] [C] [C]
    //   Absolute evict pos:              3
    //
    // Returns the overall shift in the KV Cache after eviction:
    //  - Left shrink results in no overall shift.
    //  - Right shrink shifts the cache to the right by the eviction amount.
    //  - Adaptive shrink results in shifts somewhere (inclusive) in-between left and right shrink.
    size_t evictCacheAbs(const std::vector<size_t>& absolutePositions,
                         const CacheShrinkMode shrinkMode);

    // Preserve the tokens in the cache according to the relative positions per layer per head:
    // (num_layers, num_kv_heads, preserve_positions).
    // NOTE: Adaptive shrink mode is not yet supported.
    //
    // Returns the overall shift in the KV Cache after eviction:
    //  - Left shrink results in no overall shift.
    //  - Right shrink shifts the cache to the right by the eviction amount.
    //  - Adaptive shrink: Not Supported.
    size_t preserveCache(const NestedVector<int, 3>& positions, const CacheShrinkMode shrinkMode);

protected:
    // KV Cache
    std::vector<KVCacheInfo> mKvCaches;

    // Attention Weights
    std::vector<std::string_view> mAttnWeightsBuffers;
    std::array<uint32_t, 4> mAttnWeightsShape;

    // General Context
    size_t mTokenSize = 0;
    size_t mCacheSize = 0;
    size_t mNumCachedTokens = 0;
    size_t mLeftPadSize = 0;
    size_t mRightPadSize = 0;

    // A common thread pool usable by subclasses
    ThreadPool mThreadPool;
};

namespace {
class LocalSnapKVContext;
class GlobalSnapKVContext;
} // namespace

class LocalSnapKVAgent : public CacheEvictionAgent {
public:
    explicit LocalSnapKVAgent(const std::vector<KVCacheInfo>& kvCaches, const size_t attnSinkSize,
                              const size_t numAttnHeads, const bool useIngraphAttnReduce,
                              const CacheShrinkMode shrinkMode = CacheShrinkMode::Right);

    ~LocalSnapKVAgent() override;

    // Run LocalSnapKV main algorithm
    CacheEvictionStatus run() override;

    void reset() override;

    // Requires to run before KV cache is updated
    bool requiresUpdatedKVCache() const override { return false; }

private:
    std::unique_ptr<LocalSnapKVContext> mCtx;
};

class GlobalSnapKVAgent : public CacheEvictionAgent {
public:
    explicit GlobalSnapKVAgent(const std::vector<KVCacheInfo>& kvCaches, const size_t windowSize,
                               const size_t kernelSize, const size_t maxPromptCapacity,
                               const std::string& poolingMethod,
                               const CacheShrinkMode shrinkMode = CacheShrinkMode::Right);

    ~GlobalSnapKVAgent() override;

    // Run GlobalSnapKV main algorithm
    CacheEvictionStatus run() override;

    void reset() override;

    // Requires to run after KV cache is updated
    bool requiresUpdatedKVCache() const override { return true; }

private:
    std::unique_ptr<GlobalSnapKVContext> mCtx;
};

} // namespace mtk::llm_helper
