#pragma once

#include "common/logging.h"
#include "executor/allocator.h"
#include "executor/utils/ordered_kinds.h"

namespace mtk {

namespace executor_utils {

// A wrapper for OrderedKinds of inputs and outputs that provides APIs to interact with the IOs
// and manage the indexing via the relative ordering between IOKinds.
class MutableIO {
public:
    //===--------------------===//
    // General Public IO Access
    //===--------------------===//

    // Get the number of inputs.
    size_t getNumInputs() const { return mInputs.size(); }

    // Get the number of outputs.
    size_t getNumOutputs() const { return mOutputs.size(); }

    // Get input by index.
    IOBuffer& getInput(const size_t index = 0) {
        CHECK_LT(index, getNumInputs()) << "getInput(): Index out of range.";
        return mInputs.at(index);
    }

    // Get input by index.
    const IOBuffer& getInput(const size_t index = 0) const {
        CHECK_LT(index, getNumInputs()) << "getInput(): Index out of range.";
        return mInputs.at(index);
    }

    // Get output by index.
    IOBuffer& getOutput(const size_t index = 0) {
        CHECK_LT(index, getNumOutputs()) << "getOutput(): Index out of range.";
        return mOutputs.at(index);
    }

    // Get output by index.
    const IOBuffer& getOutput(const size_t index = 0) const {
        CHECK_LT(index, getNumOutputs()) << "getOutput(): Index out of range.";
        return mOutputs.at(index);
    }

protected:
    //===------------------===//
    // IO Ordering Definition
    //===------------------===//

    // Define a new input kind with `count` number of inputs if not yet defined,
    // and append it to the back of the input ordering sequence.
    template <typename IOKindType>
    void defineInput(const IOKindType kind, const size_t count = 1) {
        mInputs.defineAppend(kind, count);
        invalidateInputIdxsCache();
    }

    // Define a new output kind with `count` number of outputs if not yet defined,
    // and append it to the back of the output ordering sequence.
    template <typename IOKindType>
    void defineOutput(const IOKindType kind, const size_t count = 1) {
        mOutputs.defineAppend(kind, count);
        invalidateOutputIdxsCache();
    }

    // Check if an input kind has been defined, whether or not it is active.
    template <typename IOKindType>
    bool isInputDefined(const IOKindType kind) const {
        return mInputs.isDefined(kind);
    }

    // Check if an output kind has been defined, whether or not it is active.
    template <typename IOKindType>
    bool isOutputDefined(const IOKindType kind) const {
        return mOutputs.isDefined(kind);
    }

    // Check if an input kind is currently accessible (via index).
    template <typename IOKindType>
    bool hasInput(const IOKindType kind) const {
        return mInputs.isOrdered(kind);
    }

    // Check if an output kind is currently accessible (via index).
    template <typename IOKindType>
    bool hasOutput(const IOKindType kind) const {
        return mOutputs.isOrdered(kind);
    }

    // Reset the current input ordering.
    void resetInputOrder() {
        DCHECK(!isInputGeneric()) << "Input ordering reset is not allowed for generic inputs.";
        mInputs.orderReset();
        invalidateInputIdxsCache();
    }

    // Reset the current output ordering.
    void resetOutputOrder() {
        DCHECK(!isOutputGeneric()) << "Output ordering reset is not allowed for generic outputs.";
        mOutputs.orderReset();
        invalidateOutputIdxsCache();
    }

    //===---------------------===//
    // Uncategorized IO Ordering
    //===---------------------===//

    // Define `count` number of generic inputs without any categorization of input kinds.
    // The input ordering sequence
    void setNumGenericInputs(const size_t count) {
        DCHECK(!mInputs.isCategorized());
        mInputs.defineGeneric(count);
    }

    // Define `count` number of generic outputs without any categorization of output kinds.
    void setNumGenericOutputs(const size_t count) {
        DCHECK(!mOutputs.isCategorized());
        mOutputs.defineGeneric(count);
    }

    // Check if the input is set to generic.
    bool isInputGeneric() const { return mInputs.isGeneric(); }

    // Check if the output is set to generic.
    bool isOutputGeneric() const { return mOutputs.isGeneric(); }

    //===---------------===//
    // IO Indexing by Kind
    //===---------------===//

    template <typename IOKindType>
    size_t getInputIndex(const IOKindType kind, const size_t pos = 0) const {
        return mInputs.getIndex(kind, pos);
    }

    template <typename IOKindType>
    size_t getOutputIndex(const IOKindType kind, const size_t pos = 0) const {
        return mOutputs.getIndex(kind, pos);
    }

    template <typename IOKindType>
    const std::vector<size_t>& getInputIndexes(const IOKindType kind) const {
        const auto kindValue = static_cast<OrderedKindBaseType>(kind);
        if (mInputIndexesCache.find(kindValue) == mInputIndexesCache.end()) {
            mInputIndexesCache.emplace(kindValue, mInputs.getIndexes(kind));
        }
        DCHECK_EQ(mInputIndexesCache.at(kindValue), mInputs.getIndexes(kind));
        return mInputIndexesCache.at(kindValue);
    }

    template <typename IOKindType>
    const std::vector<size_t>& getOutputIndexes(const IOKindType kind) const {
        const auto kindValue = static_cast<OrderedKindBaseType>(kind);
        if (mOutputIndexesCache.find(kindValue) == mOutputIndexesCache.end()) {
            mOutputIndexesCache.emplace(kindValue, mOutputs.getIndexes(kind));
        }
        DCHECK_EQ(mOutputIndexesCache.at(kindValue), mOutputs.getIndexes(kind));
        return mOutputIndexesCache.at(kindValue);
    }

    //===-----------===
    // Per-Kind Access
    //===-----------===

    // Get the number of inputs for a kind that is currently accessible via index.
    template <typename IOKindType>
    size_t getNumInputsFor(const IOKindType kind) const {
        return mInputs.getNumOrderedValues(kind);
    }

    // Get the number of outputs for a kind that is currently accessible via index.
    template <typename IOKindType>
    size_t getNumOutputsFor(const IOKindType kind) const {
        return mOutputs.getNumOrderedValues(kind);
    }

    // Get the number of inputs defined for a kind, whether or not it is accessible via index.
    template <typename IOKindType>
    size_t getNumDefinedInputsFor(const IOKindType kind) const {
        return mInputs.getNumDefinedValues(kind);
    }

    // Get the number of outputs defined for a kind, whether or not it is accessible via index.
    template <typename IOKindType>
    size_t getNumDefinedOutputsFor(const IOKindType kind) const {
        return mOutputs.getNumDefinedValues(kind);
    }

    // Get the input for a kind that was defined, whether or not it is accessible via index.
    template <typename IOKindType>
    IOBuffer& getInputFor(const IOKindType kind, const size_t pos = 0) {
        return mInputs.get(kind, pos);
    }

    // Get the input for a kind that was defined, whether or not it is accessible via index.
    template <typename IOKindType>
    const IOBuffer& getInputFor(const IOKindType kind, const size_t pos = 0) const {
        return mInputs.get(kind, pos);
    }

    // Get the output for a kind that was defined, whether or not it is accessible via index.
    template <typename IOKindType>
    IOBuffer& getOutputFor(const IOKindType kind, const size_t pos = 0) {
        return mOutputs.get(kind, pos);
    }

    // Get the output for a kind that was defined, whether or not it is accessible via index.
    template <typename IOKindType>
    const IOBuffer& getOutputFor(const IOKindType kind, const size_t pos = 0) const {
        return mOutputs.get(kind, pos);
    }

private:
    void invalidateInputIdxsCache() { mInputIndexesCache.clear(); }
    void invalidateOutputIdxsCache() { mOutputIndexesCache.clear(); }

private:
    mutable std::unordered_map<OrderedKindBaseType, std::vector<size_t>> mInputIndexesCache;
    mutable std::unordered_map<OrderedKindBaseType, std::vector<size_t>> mOutputIndexesCache;

protected:
    OrderedKinds<IOBuffer> mInputs;
    OrderedKinds<IOBuffer> mOutputs;
};

} // namespace executor_utils

} // namespace mtk