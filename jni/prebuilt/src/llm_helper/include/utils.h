#pragma once

#include <algorithm>
#include <iterator>
#include <numeric>
#include <thread>
#include <vector>

namespace mtk::llm_helper {

template <class T>
using IterValueType = typename std::iterator_traits<T>::value_type;

template <class T>
using ContainerValueType = typename std::decay_t<T>::value_type;

// Sum
template <class Iterable, class T = ContainerValueType<Iterable>>
inline T reduce_sum(const Iterable& vals) {
    return std::reduce(vals.cbegin(), vals.cend());
}

template <class InputIt, class T = IterValueType<InputIt>>
inline T reduce_sum(InputIt first, InputIt last) {
    return std::reduce(first, last);
}

// Product
template <class Iterable, class T = ContainerValueType<Iterable>>
inline std::decay_t<T> reduce_prod(const Iterable& vals, T&& init = 1) {
    return std::reduce(vals.cbegin(), vals.cend(), std::forward<T>(init), std::multiplies<>());
}

template <class InputIt, class T = IterValueType<InputIt>>
inline std::decay_t<T> reduce_prod(InputIt first, InputIt last, T&& init = 1) {
    return std::reduce(first, last, std::forward<T>(init), std::multiplies<>());
}

template <class Iterable>
inline bool allSame(const Iterable& vals) {
    auto first = vals.cbegin();
    auto last = vals.cend();
    return (std::adjacent_find(first, last, std::not_equal_to<>()) == last);
}

template <class Iterable, class UnaryOp>
inline bool allSame(const Iterable& vals, UnaryOp func) {
    auto first = vals.cbegin();
    auto last = vals.cend();

    if (first == last)
        return true;

    const auto& ref = func(*first);

    auto cur = first;
    while (++cur != last) {
        if (func(*cur) != ref)
            return false;
    }
    return true;
}

template <typename T, size_t Rank>
struct NestedVectorImpl {
    using type = std::vector<typename NestedVectorImpl<T, Rank - 1>::type>;
};

template <typename T>
struct NestedVectorImpl<T, 0> {
    using type = T;
};

// Wrapper to define nested vector.
// Example: NestedVector<float, 2> equals to std::vector<std::vector<float>>.
template <typename T, size_t Rank>
using NestedVector = typename NestedVectorImpl<T, Rank>::type;

// Data Alignment

template <size_t kAlignment, typename T>
inline constexpr T alignMultipleOf(const T value) {
    // Equivalent to: ceil(value / kAlignment) * kAlignment
    return ((value + kAlignment - 1) / kAlignment) * kAlignment;
}

// Some platforms use alignment of 16 bytes, some use 32 bytes.
#ifdef DIM_C_ALIGNMENT_BYTES
static constexpr size_t kDimCAlignmentBytesPrecise = DIM_C_ALIGNMENT_BYTES;
static constexpr size_t kDimCAlignmentBytesUpperBound = DIM_C_ALIGNMENT_BYTES;
#else
static constexpr size_t kDimCAlignmentBytesPrecise = 0UL;     // Unable to define precise alignment
static constexpr size_t kDimCAlignmentBytesUpperBound = 32UL; // Set upperbound to 32 bytes
#endif

// Precise alignment requires DIM_C_ALIGNMENT_BYTES to be defined.
template <size_t kAlignmentPrecise = kDimCAlignmentBytesPrecise>
inline constexpr size_t alignHWDimCPrecise(const size_t dimSize, const size_t typeSize) {
    static_assert(kAlignmentPrecise > 0UL, "Precise alignment is not defined.");
    return alignMultipleOf<kAlignmentPrecise>(dimSize * typeSize) / typeSize;
}

// Upper bound alignment even if DIM_C_ALIGNMENT_BYTES is not defined.
inline constexpr size_t alignHWDimCUpperBound(const size_t dimSize, const size_t typeSize) {
    return alignMultipleOf<kDimCAlignmentBytesUpperBound>(dimSize * typeSize) / typeSize;
}

} // namespace mtk::llm_helper