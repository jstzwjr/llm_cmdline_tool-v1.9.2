/* Copyright Statement:
 *
 * This software/firmware and related documentation ("MediaTek Software") are
 * protected under relevant copyright laws. The information contained herein
 * is confidential and proprietary to MediaTek Inc. and/or its licensors.
 * Without the prior written permission of MediaTek inc. and/or its licensors,
 * any reproduction, modification, use or disclosure of MediaTek Software,
 * and information contained herein, in whole or in part, shall be strictly prohibited.
 */
/* MediaTek Inc. (C) 2025. All rights reserved.
 *
 * BY OPENING THIS FILE, RECEIVER HEREBY UNEQUIVOCALLY ACKNOWLEDGES AND AGREES
 * THAT THE SOFTWARE/FIRMWARE AND ITS DOCUMENTATIONS ("MEDIATEK SOFTWARE")
 * RECEIVED FROM MEDIATEK AND/OR ITS REPRESENTATIVES ARE PROVIDED TO RECEIVER ON
 * AN "AS-IS" BASIS ONLY. MEDIATEK EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE OR NONINFRINGEMENT.
 * NEITHER DOES MEDIATEK PROVIDE ANY WARRANTY WHATSOEVER WITH RESPECT TO THE
 * SOFTWARE OF ANY THIRD PARTY WHICH MAY BE USED BY, INCORPORATED IN, OR
 * SUPPLIED WITH THE MEDIATEK SOFTWARE, AND RECEIVER AGREES TO LOOK ONLY TO SUCH
 * THIRD PARTY FOR ANY WARRANTY CLAIM RELATING THERETO. RECEIVER EXPRESSLY ACKNOWLEDGES
 * THAT IT IS RECEIVER'S SOLE RESPONSIBILITY TO OBTAIN FROM ANY THIRD PARTY ALL PROPER LICENSES
 * CONTAINED IN MEDIATEK SOFTWARE. MEDIATEK SHALL ALSO NOT BE RESPONSIBLE FOR ANY MEDIATEK
 * SOFTWARE RELEASES MADE TO RECEIVER'S SPECIFICATION OR TO CONFORM TO A PARTICULAR
 * STANDARD OR OPEN FORUM. RECEIVER'S SOLE AND EXCLUSIVE REMEDY AND MEDIATEK'S ENTIRE AND
 * CUMULATIVE LIABILITY WITH RESPECT TO THE MEDIATEK SOFTWARE RELEASED HEREUNDER WILL BE,
 * AT MEDIATEK'S OPTION, TO REVISE OR REPLACE THE MEDIATEK SOFTWARE AT ISSUE,
 * OR REFUND ANY SOFTWARE LICENSE FEES OR SERVICE CHARGE PAID BY RECEIVER TO
 * MEDIATEK FOR SUCH MEDIATEK SOFTWARE AT ISSUE.
 *
 * The following software/firmware and/or related documentation ("MediaTek Software")
 * have been modified by MediaTek Inc. All revisions are subject to any receiver's
 * applicable license agreements with MediaTek Inc.
 */

#pragma once

#include "NeuronAdapter.h"

#include <android/log.h>
#include <dlfcn.h>

#include <array>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

constexpr std::array<const char*, 4> kDefaultAdapterLib = {
    "libneuronusdk_adapter.9.mtk.so",
    "libneuronusdk_adapter.mtk.so",
    "libneuronusdk_adapter.so",
    "libneuron_adapter_mgvi.so",
};

class NeuronAdapterLib {
public:
    static std::array<const char*, 4> GetDefaultLibName() { return kDefaultAdapterLib; }

    static std::shared_ptr<NeuronAdapterLib> Create(std::string path) {
        auto lib = std::shared_ptr<NeuronAdapterLib>(new NeuronAdapterLib(path));
        return lib->IsValid() ? lib : nullptr;
    }

    static bool SetCustomLibPath(std::string path) {
        if (instance_ != nullptr) {
            return false;  // Unable to set custom lib path after initialized.
        }
        custom_lib_path_ = std::move(path);
        return true;
    }

    static void Reset() {
        instance_ = nullptr;
        custom_lib_path_.clear();
    }

    static std::shared_ptr<NeuronAdapterLib> Get() {
        std::lock_guard<std::mutex> lock(mutex_);

        if (instance_ && (custom_lib_path_.empty() || instance_->GetLibPath() == custom_lib_path_)) {
            return instance_;
        }

        if (!custom_lib_path_.empty()) {
            instance_ = Create(custom_lib_path_);
        } else {
            for (const auto& path : GetDefaultLibName()) {
                instance_ = Create(path);
                if (instance_) {
                    break;
                }
            }
        }
        return instance_;
    }

    bool IsValid() const { return is_valid_; }

    std::string GetLibPath() const { return lib_path_; }

    ~NeuronAdapterLib() {
        if (handle_) {
            dlclose(handle_);
        }
    }

#define DEF_HIDDEN_SYMBOL(symbol, returnType, ...) \
public:                                            \
    template <typename... Args>                    \
    auto symbol(Args... args) -> returnType {      \
        return symbol##_(args...);                 \
    }                                              \
                                                   \
private:                                           \
    returnType (*symbol##_)(__VA_ARGS__) = nullptr;
    DEF_HIDDEN_SYMBOL(NeuronCompilation_createWithOptions, int, NeuronModel*, NeuronCompilation**,
                      const char*);
#undef DEF_HIDDEN_SYMBOL

#define DEF_SYMBOL(symbol)                                     \
public:                                                        \
    template <typename... Args>                                \
    auto symbol(Args... args) -> decltype(::symbol(args...)) { \
        return symbol##_(args...);                             \
    }                                                          \
                                                               \
private:                                                       \
    decltype(&::symbol) symbol##_ = nullptr;
    DEF_SYMBOL(Neuron_getFeatureSupportedStatus);
    DEF_SYMBOL(Neuron_getL1MemorySizeKb);
    DEF_SYMBOL(Neuron_getVersion);
    DEF_SYMBOL(NeuronMemory_createFromFd);
#ifdef __ANDROID__
    DEF_SYMBOL(NeuronMemory_createFromAHardwareBuffer);
#endif // __ANDROID__
    DEF_SYMBOL(NeuronMemory_free);
    DEF_SYMBOL(NeuronModel_create);
    DEF_SYMBOL(NeuronModel_free);
    DEF_SYMBOL(NeuronModel_finish);
    DEF_SYMBOL(NeuronModel_addOperand);
    DEF_SYMBOL(NeuronModel_setOperandValue);
    DEF_SYMBOL(NeuronModel_setOperandValueFromModel);
    DEF_SYMBOL(NeuronModel_addOperation);
    DEF_SYMBOL(NeuronModel_addOperationExtension);
    DEF_SYMBOL(NeuronModel_identifyInputsAndOutputs);
    DEF_SYMBOL(NeuronCompilation_create);
    DEF_SYMBOL(NeuronCompilation_createV2);
    DEF_SYMBOL(NeuronCompilation_free);
    DEF_SYMBOL(NeuronCompilation_finish);
    DEF_SYMBOL(NeuronCompilation_setPreference);
    DEF_SYMBOL(NeuronCompilation_setPriority);
    DEF_SYMBOL(NeuronCompilation_getInputPaddedDimensions);
    DEF_SYMBOL(NeuronCompilation_getOutputPaddedDimensions);
    DEF_SYMBOL(NeuronCompilation_getInputPaddedSize);
    DEF_SYMBOL(NeuronCompilation_getOutputPaddedSize);
    DEF_SYMBOL(NeuronCompilation_setOptimizationString);
    DEF_SYMBOL(NeuronExecution_create);
    DEF_SYMBOL(NeuronExecution_free);
    DEF_SYMBOL(NeuronExecution_setInput);
    DEF_SYMBOL(NeuronExecution_setOutput);
    DEF_SYMBOL(NeuronExecution_setInputFromMemory);
    DEF_SYMBOL(NeuronExecution_setOutputFromMemory);
    DEF_SYMBOL(NeuronExecution_compute);
    DEF_SYMBOL(NeuronExecution_setBoostHint);
    DEF_SYMBOL(NeuronExecution_setCacheFlushHint);
    DEF_SYMBOL(NeuronModel_getExtensionOperationType);
    DEF_SYMBOL(NeuronModel_getExtensionOperandType);
#undef DEF_SYMBOL

private:
    void LoadSymbol() {
#define LOAD_SYMBOL(symbol)                                                          \
    do {                                                                             \
        symbol##_ = reinterpret_cast<decltype(symbol##_)>(dlsym(handle_, #symbol));  \
        if (!symbol##_) {                                                            \
            __android_log_print(                                                     \
                ANDROID_LOG_ERROR, "AdapterLib", "Failed to load symbol: ##symbol"); \
            return;                                                                  \
        }                                                                            \
    } while (0)

        LOAD_SYMBOL(Neuron_getFeatureSupportedStatus);
        LOAD_SYMBOL(Neuron_getL1MemorySizeKb);
        LOAD_SYMBOL(Neuron_getVersion);
        LOAD_SYMBOL(NeuronMemory_createFromFd);
#ifdef __ANDROID__
        LOAD_SYMBOL(NeuronMemory_createFromAHardwareBuffer);
#endif // __ANDROID__
        LOAD_SYMBOL(NeuronMemory_free);
        LOAD_SYMBOL(NeuronModel_create);
        LOAD_SYMBOL(NeuronModel_free);
        LOAD_SYMBOL(NeuronModel_finish);
        LOAD_SYMBOL(NeuronModel_addOperand);
        LOAD_SYMBOL(NeuronModel_setOperandValue);
        LOAD_SYMBOL(NeuronModel_setOperandValueFromModel);
        LOAD_SYMBOL(NeuronModel_addOperation);
        LOAD_SYMBOL(NeuronModel_addOperationExtension);
        LOAD_SYMBOL(NeuronModel_identifyInputsAndOutputs);
        LOAD_SYMBOL(NeuronCompilation_create);
        LOAD_SYMBOL(NeuronCompilation_createV2);
        LOAD_SYMBOL(NeuronCompilation_free);
        LOAD_SYMBOL(NeuronCompilation_finish);
        LOAD_SYMBOL(NeuronCompilation_setPreference);
        LOAD_SYMBOL(NeuronCompilation_setPriority);
        LOAD_SYMBOL(NeuronCompilation_getInputPaddedDimensions);
        LOAD_SYMBOL(NeuronCompilation_getOutputPaddedDimensions);
        LOAD_SYMBOL(NeuronCompilation_getInputPaddedSize);
        LOAD_SYMBOL(NeuronCompilation_getOutputPaddedSize);
        LOAD_SYMBOL(NeuronCompilation_setOptimizationString);
        LOAD_SYMBOL(NeuronExecution_create);
        LOAD_SYMBOL(NeuronExecution_free);
        LOAD_SYMBOL(NeuronExecution_setInput);
        LOAD_SYMBOL(NeuronExecution_setOutput);
        LOAD_SYMBOL(NeuronExecution_setInputFromMemory);
        LOAD_SYMBOL(NeuronExecution_setOutputFromMemory);
        LOAD_SYMBOL(NeuronExecution_compute);
        LOAD_SYMBOL(NeuronExecution_setBoostHint);
        LOAD_SYMBOL(NeuronExecution_setCacheFlushHint);
        LOAD_SYMBOL(NeuronModel_getExtensionOperationType);
        LOAD_SYMBOL(NeuronModel_getExtensionOperandType);
        LOAD_SYMBOL(NeuronCompilation_createWithOptions);
        is_valid_ = true;
    }

private:
    explicit NeuronAdapterLib(const std::string libPath) : lib_path_(libPath) {
        handle_ = dlopen(lib_path_.c_str(), RTLD_LAZY);
        __android_log_print(
            ANDROID_LOG_INFO, "AdapterLib", "Try to open library %s", libPath.c_str());
        if (!handle_) {
            const char* dl_error = dlerror();
            __android_log_print(ANDROID_LOG_ERROR, "AdapterLib", "Unable to open library %s : %s",
                                libPath.c_str(), dl_error);
        } else {
            LoadSymbol();
        }
    }

    NeuronAdapterLib(const NeuronAdapterLib&) = delete;
    NeuronAdapterLib& operator=(const NeuronAdapterLib&) = delete;

    bool is_valid_ = false;

    std::string lib_path_;

    void* handle_ = nullptr;

    inline static std::shared_ptr<NeuronAdapterLib> instance_;

    inline static std::string custom_lib_path_;

    inline static std::mutex mutex_;
};