#include "executor/allocator.h"

#include "backend/api/neuron/NeuronAdapterLib.h"
#include "common/logging.h"

#include <android/hardware_buffer.h>
#include <fcntl.h>
#include <linux/dma-heap.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#ifdef USE_USDK_BACKEND
static constexpr bool kUseUsdkBackend = true;
#else
static constexpr bool kUseUsdkBackend = false;
#endif

namespace mtk {

Allocator::~Allocator() {
    // NOTE: Allocator::releaseAll() needs to be called before reaching this dtor.
    DCHECK_EQ(mAllocatedBuffers.size(), 0)
        << "Buffers are not released before destructing Allocator instance.";
}

IOBuffer Allocator::allocate(const size_t size, const bool disableTracking) {
    IOBuffer ioBuffer;
    ioBuffer.sizeBytes = size;
    ioBuffer.usedSizeBytes = size;
    if (!allocateMemory(ioBuffer)) {
        return IOBuffer(); // Empty buffer
    }
    if (disableTracking) {
        // Remove from allocation tracking so that releaseAll() won't release this buffer.
        std::scoped_lock lock(mMutex);
        mAllocatedBuffers.erase(ioBuffer.buffer);
    }
    return ioBuffer;
}

bool Allocator::release(void* addr) {
    std::scoped_lock lock(mMutex);
    if (mAllocatedBuffers.find(addr) == mAllocatedBuffers.end()) {
        LOG(ERROR) << "Unable to release the memory at " << addr << " by the allocator instance "
                   << this;
        return false;
    }
    auto& ioBuffer = mAllocatedBuffers.at(addr);
    return releaseMemory(ioBuffer);
}

void Allocator::releaseAll() {
    std::scoped_lock lock(mMutex);
    while (!mAllocatedBuffers.empty()) {
        auto& [addr, ioBuffer] = *mAllocatedBuffers.begin();
        releaseMemory(ioBuffer);
    }
}

std::shared_ptr<NeuronAdapterLib> Allocator::getNeuronLib() {
    if (!mNpLib) {
        mNpLib = NeuronAdapterLib::Get();
    }
    DCHECK(mNpLib);
    return mNpLib;
}

DmaBufferAllocator::~DmaBufferAllocator() {
    releaseAll();
}

bool DmaBufferAllocator::allocateMemory(IOBuffer& ioBuffer) {
    const auto& size = ioBuffer.sizeBytes;
    int fd = open("/dev/dma_heap/system-uncached", O_RDWR);
    if (fd < 0) {
        LOG(ERROR) << "Failed to open dma_heap/system-uncached";
        return false;
    }

    struct dma_heap_allocation_data heapInfo = {
        .len = size,
        .fd_flags = O_RDWR | O_CLOEXEC,
    };

    if (ioctl(fd, DMA_HEAP_IOCTL_ALLOC, &heapInfo) < 0) {
        close(fd);
        LOG(ERROR) << "Failed to allocate DMA heap memory";
        return false;
    };
    close(fd);

    void* bufferAddr = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, heapInfo.fd, 0);
    if (bufferAddr == MAP_FAILED) {
        LOG(ERROR) << "DMA buffer allocation: mmap failed!";
        return false;
    }
    ioBuffer.fd = heapInfo.fd;
    ioBuffer.buffer = bufferAddr;

    // USDK requires NeuronMemory
    if constexpr (kUseUsdkBackend) {
        getNeuronLib()->NeuronMemory_createFromFd(
            ioBuffer.sizeBytes, PROT_READ | PROT_WRITE, ioBuffer.fd, 0, &ioBuffer.neuronMemory);
    }

    // Add to allocated buffer record
    std::scoped_lock lock(mMutex);
    mAllocatedBuffers.emplace(bufferAddr, ioBuffer);
    return true;
}

bool DmaBufferAllocator::releaseMemory(IOBuffer& ioBuffer) {
    if (!ioBuffer.isAllocated()) {
        return true; // Do nothing
    }

    if (ioBuffer.neuronMemory != nullptr) {
        getNeuronLib()->NeuronMemory_free(ioBuffer.neuronMemory);
        ioBuffer.neuronMemory = nullptr;
    }

    if ((void*)::munmap(ioBuffer.buffer, ioBuffer.sizeBytes) == MAP_FAILED) {
        LOG(ERROR) << "DMA buffer allocation: munmap failed!";
        return false;
    }
    if (close(ioBuffer.fd) != 0) {
        return false;
    }

    const auto bufferAddr = ioBuffer.buffer;

    ioBuffer.buffer = nullptr;
    ioBuffer.sizeBytes = 0;
    ioBuffer.fd = -1;

    // Remove from allocated buffer record if exist
    std::scoped_lock lock(mMutex);
    mAllocatedBuffers.erase(bufferAddr);

    return true;
}

AhwBufferAllocator::~AhwBufferAllocator() {
    releaseAll();
}

bool AhwBufferAllocator::allocateMemory(IOBuffer& ioBuffer) {
    const auto& size = ioBuffer.sizeBytes;
    auto usage = AHARDWAREBUFFER_USAGE_CPU_READ_RARELY | AHARDWAREBUFFER_USAGE_CPU_WRITE_RARELY;

    AHardwareBuffer_Desc ahwbDesc = {
        .width = static_cast<uint32_t>(size),
        .height = 1,
        .layers = 1,
        .format = AHARDWAREBUFFER_FORMAT_BLOB,
        .usage = usage,
        .stride = static_cast<uint32_t>(size),
    };

    AHardwareBuffer* ahwbHandle = nullptr;
    if (AHardwareBuffer_allocate(&ahwbDesc, &ahwbHandle) != 0) {
        LOG(ERROR) << "Allocate AHardwareBuffer fail";
        return false;
    }

    void* bufferAddr = nullptr;
    AHardwareBuffer_lock(ahwbHandle, usage, -1, NULL, &bufferAddr);

    ioBuffer.ahwbHandle = ahwbHandle;
    ioBuffer.buffer = bufferAddr;

    // USDK requires NeuronMemory
    if constexpr (kUseUsdkBackend) {
        getNeuronLib()->NeuronMemory_createFromAHardwareBuffer(
            ioBuffer.ahwbHandle, &ioBuffer.neuronMemory);
    }

    // Add to allocated buffer record
    std::scoped_lock lock(mMutex);
    mAllocatedBuffers.emplace(bufferAddr, ioBuffer);
    return true;
}

bool AhwBufferAllocator::releaseMemory(IOBuffer& ioBuffer) {
    if (!ioBuffer.isAllocated()) {
        return true; // Do nothing
    }

    if (ioBuffer.neuronMemory != nullptr) {
        getNeuronLib()->NeuronMemory_free(ioBuffer.neuronMemory);
        ioBuffer.neuronMemory = nullptr;
    }

    AHardwareBuffer_unlock(ioBuffer.ahwbHandle, nullptr);
    AHardwareBuffer_release(ioBuffer.ahwbHandle);

    const auto bufferAddr = ioBuffer.buffer;

    ioBuffer.buffer = nullptr;
    ioBuffer.sizeBytes = 0;
    ioBuffer.fd = -1;
    ioBuffer.ahwbHandle = nullptr;

    // Remove from allocated buffer record if exist
    std::scoped_lock lock(mMutex);
    mAllocatedBuffers.erase(bufferAddr);

    return true;
}

} // namespace mtk