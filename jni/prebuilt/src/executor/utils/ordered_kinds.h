#pragma once

#include "common/logging.h"

#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mtk {

namespace executor_utils {

// The underlying integral type for Kind type
using OrderedKindBaseType = int32_t;

// This class manages the ordering sequence of values that are categorized according to their kinds.
// The relative ordering of values within a kind stays constant throughout their lifetime.
//
//  Example:
//  ```
//    enum class SampleKind { A, B, C, D };
//    OrderedKinds<std::string> ordStr;
//    // First define the kinds and their values:
//    ordStr.define(SampleKind::A, {"A1", "A2"});
//    ordStr.define(SampleKind::B, {"B1", "B2", "B3"});
//    ordStr.define(SampleKind::C, {"C"});
//    // Then set the relative order between the kinds:
//    ordStr.orderAppend(SampleKind::A); // {"A1", "A2"}
//    ordStr.orderAppend(SampleKind::B); // {"A1", "A2", "B1", "B2", "B3"}
//    ordStr.orderAppend(SampleKind::C); // {"A1", "A2", "B1", "B2", "B3", "C"}
//    // The ordered values are now {"A1", "A2", "B1", "B2", "B3", "C"}
//    CHECK(ordStr[0] == "A1");
//    CHECK(ordStr[1] == "A2");
//    CHECK(ordStr[2] == "B1"); // and so on ...
//
//    // Reordering and subset selection of existing defined kinds.
//    ordStr.orderReset();
//    ordStr.orderAppend(SampleKind::B); // {"B1", "B2", "B3"}
//    ordStr.orderAppend(SampleKind::A); // {"B1", "B2", "B3", "A1", "A2"}
//    // The ordered values are now {"B1", "B2", "B3", "A1", "A2"}
//    CHECK(ordStr[0] == "B1");
//    CHECK(ordStr[1] == "B2");
//    CHECK(ordStr[2] == "B3"); // and so on ...
//
//    // Reordering while introducing a new kind and overwriting existing values.
//    ordStr.orderReset();
//    ordStr.define(SampleKind::A, 2); // No change of values, ensure has two items.
//    ordStr.define(SampleKind::D, {"D0", "D1", "D2"}); // Define a new kind.
//    ordStr.define(SampleKind::B, {"B1", "B2", "B0"}, true); // Overwrite values.
//    // Set the order with a single function call:
//    ordStr.orderAssign({SampleKind::A, SampleKind::D, SampleKind::B});
//    // The ordered values are now {"A1", "A2", "D0", "D1", "D2", "B1", "B2", "B0"}
//    CHECK(ordStr[0] == "A1");
//    CHECK(ordStr[1] == "A2");
//    CHECK(ordStr[2] == "D0"); // and so on ...
//  ```
template <typename ValueType>
class OrderedKinds {
public:
    //===-----------------===/
    // Accessor and Iterator
    //===-----------------===/

    // Get the number of items that are currently in the ordering list, i.e. accessible via index.
    size_t size() const { return mOrderedValues.size(); }

    // Check if there is any ordered values accessible via index.
    bool isEmpty() const { return mOrderedValues.empty(); }

    // Get the number of defined kinds.
    size_t getNumKinds() const { return mKindToValuesMap.size(); }

    // Get the number of items defined for a kind.
    template <typename KindType>
    size_t getNumDefinedValues(const KindType kind) const;

    // Get the number of items defined for an ordered kind.
    // Returns 0 if the kind is not defined or not currently in the ordered list,
    // otherwise returns the number of items defined for this kind.
    template <typename KindType>
    size_t getNumOrderedValues(const KindType kind) const;

    // Check if only generic kind is defined and there is no custom kind defined.
    bool isGeneric() const {
        return (mKindToValuesMap.size() == 1) && (mKindToValuesMap.begin()->first == kGenericKind);
    }

    // Check if there is at least one custom kind defined.
    bool isCategorized() const { return !isEmpty() && !isGeneric(); }

    // Check if a kind has been defined, whether or not it is currently ordered.
    template <typename KindType>
    bool isDefined(const KindType kind) const;

    // Check if a kind is defined and is currently in the ordering list, i.e. accessible via index.
    template <typename KindType>
    bool isOrdered(const KindType kind) const;

    auto begin() { return mOrderedValues.begin(); }

    auto begin() const { return mOrderedValues.begin(); }

    auto end() { return mOrderedValues.end(); }

    auto end() const { return mOrderedValues.end(); }

    //===--------===//
    // Value Access
    //===--------===//

    // Get item via index.
    ValueType& operator[](const size_t index) { return at(index); }

    // Get item via index.
    ValueType& at(const size_t index);

    // Get item via index.
    const ValueType& at(const size_t index) const;

    // Get the item at a specific position within a kind, bypassing the ordering between kinds.
    template <typename KindType>
    ValueType& get(const KindType kind, const size_t pos = 0);

    // Get the item at a specific position within a kind, bypassing the ordering between kinds.
    template <typename KindType>
    const ValueType& get(const KindType kind, const size_t pos = 0) const;

    // Get all values in a kind, bypassing the ordering between kinds.
    template <typename KindType>
    std::vector<ValueType>& getValues(const KindType kind);

    // Get all values in a kind, bypassing the ordering between kinds.
    template <typename KindType>
    const std::vector<ValueType>& getValues(const KindType kind) const;

    // Get the index of the n-th position value of an active kind based on the current ordering.
    template <typename KindType>
    size_t getIndex(const KindType kind, const size_t pos = 0) const;

    // Get the indexes of all values of an active kind based on the current ordering.
    template <typename KindType>
    std::vector<size_t> getIndexes(const KindType kind) const;

    // Get the current ordering of kinds.
    template <typename KindType = OrderedKindBaseType>
    std::vector<KindType> getKindOrdering() const;

    //===-------------------===//
    // Ordered Kind Definition
    //===-------------------===//

    // Define a new ordered kind if not yet defined, where the values will be default constructed.
    // If the `kind` has already been defined previously, then the `count` param must match the
    // existing count.
    template <typename KindType>
    void define(const KindType kind, const size_t count = 1);

    // Define a new ordered kind with values to initialize if not yet defined.
    // If `allowOverwrite` is set to true, then any existing values will be overwritten with the
    // provided `values`.
    template <typename KindType>
    void define(const KindType kind, std::vector<ValueType> values,
                const bool allowOverwrite = false);

    // Define a kind then append to the back of the ordering sequence list.
    template <typename KindType>
    void defineAppend(const KindType kind, const size_t count = 1) {
        if (count == 0) {
            return;
        }
        define(kind, count);
        orderAppend(kind);
    }

    // Define a kind then append to the back of the ordering sequence list.
    template <typename KindType>
    void defineAppend(const KindType kind, std::vector<ValueType> values,
                      const bool allowOverwrite = false) {
        if (values.empty()) {
            return;
        }
        define(kind, std::move(values), allowOverwrite);
        orderAppend(kind);
    }

    // Define a single generic kind ordering.
    void defineGeneric(const size_t count) { defineAppend(kGenericKind, count); }

    // Define a single generic kind ordering with values to initialize.
    void defineGeneric(std::vector<ValueType> values, const bool allowOverwrite = false) {
        defineAppend(kGenericKind, values, allowOverwrite);
    }

    // Clear all definitions and their values.
    void clear();

    //===---------------===//
    // Ordering Management
    //===---------------===//

    // Append an existing defined kind to the back of the ordering sequence list.
    template <typename KindType>
    void orderAppend(const KindType kind);

    // Assign the ordering of defined kinds according to a given ordering sequence list.
    template <typename KindType>
    void orderAssign(const std::vector<KindType>& kindOrdering);

    // Rebuild the ordered value references.
    void orderRebuild();

    // Invalidate the ordering between kinds. All values of existing defined kinds are intact.
    void orderReset();

private:
    static constexpr OrderedKindBaseType kGenericKind = -1;

    // A mapping of defined kinds to each of their items.
    std::unordered_map<OrderedKindBaseType, std::vector<ValueType>> mKindToValuesMap;

    // The current ordering of kinds.
    std::vector<OrderedKindBaseType> mKindOrdering;

    // Ordered references to items based on `mKindOrdering`
    std::vector<ValueType*> mOrderedValues;

    // A mapping of currently active kinds to each of their starting indexes in ordered items list.
    // Used for quick index lookup.
    std::unordered_map<OrderedKindBaseType, size_t> mOrderedKindStartIdxMap;
};

template <typename ValueType>
template <typename KindType>
size_t OrderedKinds<ValueType>::getNumDefinedValues(const KindType kind) const {
    if (!isDefined(kind)) {
        return 0;
    }
    return mKindToValuesMap.at(static_cast<OrderedKindBaseType>(kind)).size();
}

template <typename ValueType>
template <typename KindType>
size_t OrderedKinds<ValueType>::getNumOrderedValues(const KindType kind) const {
    if (!isOrdered(kind)) {
        return 0;
    }
    DCHECK(isDefined(kind));
    return getNumDefinedValues(kind);
}

template <typename ValueType>
template <typename KindType>
bool OrderedKinds<ValueType>::isDefined(const KindType kind) const {
    const auto kindValue = static_cast<OrderedKindBaseType>(kind);
    return mKindToValuesMap.find(kindValue) != mKindToValuesMap.end()
           && !mKindToValuesMap.at(kindValue).empty();
}

template <typename ValueType>
template <typename KindType>
bool OrderedKinds<ValueType>::isOrdered(const KindType kind) const {
    const auto kindValue = static_cast<OrderedKindBaseType>(kind);
    return mOrderedKindStartIdxMap.find(kindValue) != mOrderedKindStartIdxMap.end();
}

template <typename ValueType>
ValueType& OrderedKinds<ValueType>::at(const size_t index) {
    DCHECK_LT(index, size()) << "at(): Index out of range.";
    return *mOrderedValues[index];
}

template <typename ValueType>
const ValueType& OrderedKinds<ValueType>::at(const size_t index) const {
    DCHECK_LT(index, size()) << "at(): Index out of range.";
    return *mOrderedValues[index];
}

template <typename ValueType>
template <typename KindType>
ValueType& OrderedKinds<ValueType>::get(const KindType kind, const size_t pos) {
    const auto targetKind = static_cast<OrderedKindBaseType>(kind);
    DCHECK(isDefined(targetKind)) << targetKind;
    DCHECK_LT(pos, mKindToValuesMap.at(targetKind).size())
        << "get(): Kind (" << targetKind << ") position out of range: " << pos;
    return mKindToValuesMap.at(targetKind)[pos];
}

template <typename ValueType>
template <typename KindType>
const ValueType& OrderedKinds<ValueType>::get(const KindType kind, const size_t pos) const {
    return const_cast<OrderedKinds*>(this)->get(kind, pos);
}

template <typename ValueType>
template <typename KindType>
std::vector<ValueType>& OrderedKinds<ValueType>::getValues(const KindType kind) {
    const auto targetKind = static_cast<OrderedKindBaseType>(kind);
    DCHECK(isDefined(targetKind)) << targetKind;
    return mKindToValuesMap.at(targetKind);
}

template <typename ValueType>
template <typename KindType>
const std::vector<ValueType>& OrderedKinds<ValueType>::getValues(const KindType kind) const {
    return const_cast<OrderedKinds*>(this)->getValues(kind);
}

template <typename ValueType>
template <typename KindType>
size_t OrderedKinds<ValueType>::getIndex(const KindType kind, const size_t pos) const {
    const auto targetKind = static_cast<OrderedKindBaseType>(kind);
    DCHECK(isDefined(targetKind)) << targetKind;
    DCHECK(isOrdered(targetKind)) << "No index for inactive kind: " << targetKind;

    const auto startIdx = mOrderedKindStartIdxMap.at(targetKind);
    const auto numItems = mKindToValuesMap.at(targetKind).size();
    DCHECK_LT(pos, numItems) << "Kind pos out of range.";
    return startIdx + pos;
}

template <typename ValueType>
template <typename KindType>
std::vector<size_t> OrderedKinds<ValueType>::getIndexes(const KindType kind) const {
    if (!isOrdered(kind)) {
        return {};
    }
    const auto targetKind = static_cast<OrderedKindBaseType>(kind);
    const auto startIdx = mOrderedKindStartIdxMap.at(targetKind);
    const auto numItems = mKindToValuesMap.at(targetKind).size();

    std::vector<size_t> indexes;
    indexes.resize(numItems);
    std::iota(indexes.begin(), indexes.end(), startIdx);
    DCHECK(!indexes.empty());
    return indexes;
}

template <typename ValueType>
template <typename KindType>
std::vector<KindType> OrderedKinds<ValueType>::getKindOrdering() const {
    std::vector<KindType> kindOrdering;
    kindOrdering.reserve(mKindOrdering.size());
    std::transform(mKindOrdering.begin(), mKindOrdering.end(), std::back_inserter(kindOrdering),
                   [](const auto kind) { return static_cast<KindType>(kind); });
    return kindOrdering;
}

template <typename ValueType>
template <typename KindType>
void OrderedKinds<ValueType>::define(const KindType kind, const size_t count) {
    if (count == 0) {
        return; // Skip
    }
    const auto kindValue = static_cast<OrderedKindBaseType>(kind);

    // Checks
    DCHECK(!isDefined(kind) || getNumDefinedValues(kind) == count)
        << "Trying to overwrite an existing kind (" << kindValue << ") having "
        << getNumDefinedValues(kind) << " item(s) to " << count;

    // Add kind to map if not defined before,
    // so no change if kind already existed and count is the same.
    mKindToValuesMap[kindValue].resize(count);

    CHECK(!isDefined(kGenericKind) || mKindToValuesMap.size() == 1)
        << "A mixture of generic and custom kind is not allowed.";
}

template <typename ValueType>
template <typename KindType>
void OrderedKinds<ValueType>::define(const KindType kind, std::vector<ValueType> values,
                                     const bool allowOverwrite) {
    if (values.empty()) {
        return; // Skip
    }
    const auto kindValue = static_cast<OrderedKindBaseType>(kind);
    const auto count = values.size();

    // Allow overwriting of values only when the number of values stays the same.
    bool isValueModified = false;
    if (isDefined(kind)) {
        const auto numDefinedValues = getNumDefinedValues(kind);
        const auto& existingValues = mKindToValuesMap.at(kindValue);
        if (allowOverwrite) {
            isValueModified = (numDefinedValues != count || existingValues != values);
        } else {
            if (numDefinedValues != count) {
                LOG(ERROR) << "Trying to overwrite an existing kind (" << kindValue << ") having "
                           << numDefinedValues << " item(s) to " << count;
                return;
            }
            if (existingValues != values) {
                LOG(ERROR) << "Trying to overwrite an existing kind (" << kindValue
                           << ") with different values.";
                return;
            }
        }
    }

    // Add kind to map if not defined before, otherwise overwrite the values.
    mKindToValuesMap.insert_or_assign(kindValue, std::move(values));

    // Rebuild the values ordering references if values are modified.
    if (isValueModified) {
        orderRebuild();
    }

    CHECK(!isDefined(kGenericKind) || mKindToValuesMap.size() == 1)
        << "A mixture of generic and custom kind is not allowed.";
}

template <typename ValueType>
template <typename KindType>
void OrderedKinds<ValueType>::orderAppend(const KindType kind) {
    const auto kindValue = static_cast<OrderedKindBaseType>(kind);
    CHECK(isDefined(kind)) << "Appending an undefined kind " << kindValue;
    mKindOrdering.push_back(kindValue);

    const size_t startIdx = mOrderedValues.size();
    auto& values = mKindToValuesMap.at(kindValue);
    for (auto& value : values) {
        mOrderedValues.push_back(&value);
    }
    DCHECK(mOrderedKindStartIdxMap.find(kindValue) == mOrderedKindStartIdxMap.end())
        << "Duplicated kind detected in the kind ordering sequence list: " << kindValue;
    mOrderedKindStartIdxMap.emplace(kindValue, startIdx); // Mark kind as active with start idx
}

template <typename ValueType>
template <typename KindType>
void OrderedKinds<ValueType>::orderAssign(const std::vector<KindType>& kindOrdering) {
    orderReset();
    for (const auto kind : kindOrdering) {
        DCHECK(isDefined(kind));
        orderAppend(kind);
    }
}

template <typename ValueType>
void OrderedKinds<ValueType>::orderRebuild() {
    const auto kindOrdering = mKindOrdering; // Make a copy
    orderReset();
    orderAssign(kindOrdering);
}

template <typename ValueType>
void OrderedKinds<ValueType>::orderReset() {
    mKindOrdering.clear();
    mOrderedKindStartIdxMap.clear();
    mOrderedValues.clear();
}

template <typename ValueType>
void OrderedKinds<ValueType>::clear() {
    mKindToValuesMap.clear();
    orderReset();
}

} // namespace executor_utils

} // namespace mtk