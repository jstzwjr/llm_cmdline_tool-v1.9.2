#pragma once

#include "logging.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#ifdef NUM_THREAD_WORKERS
static const size_t kDefaultNumThreadWorkers = NUM_THREAD_WORKERS;
#else
static const size_t kDefaultNumThreadWorkers = std::min(2U, std::thread::hardware_concurrency());
#endif

// A thread pool that only ensures there are at most N number of active threads in the pool,
// without any regard to thread creation/destruction overhead.
// However, if `numWorkers` is set to 0, then there is no limit to the number of active threads.
class BasicThreadPool {
public:
    BasicThreadPool(const size_t numWorkers) : kMaxNumWorkers(numWorkers) {}

    BasicThreadPool() : kMaxNumWorkers(kDefaultNumThreadWorkers) {}

    ~BasicThreadPool() {
        // Ensure all threads are joined before destroying itself
        joinAll();
    }

    bool empty() const { return mThreadPool.empty(); }

    template <class Func, class... Args>
    void push(Func&& func, Args&&... args) {
        // Create a new thread and start executing it
        mThreadPool.emplace_back(std::forward<Func>(func), std::forward<Args>(args)...);

        // Join the threads once the number of active threads have reached `kMaxNumWorkers`.
        // However if `kMaxNumWorkers` is 0, allow unlimited number of active threads.
        if (kMaxNumWorkers && mThreadPool.size() == kMaxNumWorkers)
            joinAll();
    }

    void wait() { joinAll(); }

    void joinAll() {
        for (auto& thread : mThreadPool)
            thread.join();
        mThreadPool.clear();
    }

private:
    const size_t kMaxNumWorkers;
    std::deque<std::thread> mThreadPool;
};

enum class ThreadSpawn {
    All,    // Spawn all threads upfront.
    Linear, // Spawn threads linearly on demand. Example: 1, 2, 3, 4, ...
    Exp     // Spawn threads exponentially on demand. Example: 1, 2, 4, 8, ...
};

// A thread pool where each thread worker is in a loop waiting to run a task in the backlog.
// If `numWorkers` is given, then there can only be at most `numWorkers` running at the background,
// otherwise the max number of workers will be initialized to `kDefaultNumThreadWorkers`.
class ThreadPool {
public:
    ThreadPool(const size_t numWorkers, const ThreadSpawn spawnPolicy = ThreadSpawn::All)
        : kMaxNumWorkers(numWorkers), kSpawnPolicy(spawnPolicy) {
        spawnWorkers();
    }

    ThreadPool(const ThreadSpawn spawnPolicy = ThreadSpawn::All)
        : kMaxNumWorkers(kDefaultNumThreadWorkers), kSpawnPolicy(spawnPolicy) {
        spawnWorkers();
    }

    ~ThreadPool() {
        // Ensure all threads are joined before destroying itself
        joinAll();
    }

    // Check if there is at least one task currently being run
    bool hasRunningTask() const { return mNumActiveWorkers > 0; }

    // Push a task to the thread pool and return once the task has started to run by a thread worker
    template <class Func, class... Args>
    void push(Func&& func, Args&&... args) {
        if (needMoreWorkers()) {
            std::scoped_lock lock(mUserMutex);
            spawnWorkers();
        }

        std::unique_lock lock(mTaskMutex);
        mTaskBacklog.push_back(std::bind(std::forward<Func>(func), std::forward<Args>(args)...));

        lock.unlock();
        mCvNewTask.notify_one();
    }

    // Wait for tasks to finish
    void wait() {
        if (mWorkers.empty()) {
            return; // Nothing to wait
        }
        // Wait until no more active worker threads and task backlog is empty
        std::unique_lock lock(mTaskMutex);
        mCvActiveWorkers.wait(lock, [this] { return !hasTodoTask() && !hasRunningTask(); });
    }

    // Join all threads and terminate them
    void joinAll() {
        if (mWorkers.empty()) {
            return; // Nothing to join
        }

        // Wait for all tasks to be finished
        wait();

        std::scoped_lock lock(mUserMutex);

        // Send terminate signal to all thread workers
        std::unique_lock taskLock(mTaskMutex);
        mTerminateSignal = true;
        taskLock.unlock();
        mCvNewTask.notify_all();

        // Join all thread workers
        for (auto& worker : mWorkers) {
            worker.join();
        }
        mWorkers.clear();

        // Reset terminate signal
        mTerminateSignal = false;
    }

private:
    // Check if there is any task in the backlog waiting to run
    bool hasTodoTask() const { return !mTaskBacklog.empty(); }

    // Check if there is any idle worker threads
    bool needMoreWorkers() const {
        const auto curNumWorkers = mWorkers.size();
        if (curNumWorkers >= kMaxNumWorkers) {
            return false; // Already fully spawned
        }
        DCHECK_GE(curNumWorkers, mNumActiveWorkers);
        const auto numIdleWorkers = curNumWorkers - mNumActiveWorkers;
        return numIdleWorkers == 0; // Spawn more workers if no more idle workers
    }

    // Spawn new workers if not fully spawned
    void spawnWorkers() {
        CHECK_GE(kMaxNumWorkers, 1) << "The number of thread workers must be >= 1";

        const auto curNumWorkers = mWorkers.size();
        DCHECK_LE(curNumWorkers, kMaxNumWorkers);

        if (curNumWorkers == kMaxNumWorkers) {
            return; // Already fully spawned
        }

        const size_t newWorkerCount = [this, curNumWorkers]() -> size_t {
            switch (kSpawnPolicy) {
                case ThreadSpawn::All:
                    DCHECK(mWorkers.empty()) << "Thread workers have already spawned.";
                    return kMaxNumWorkers;
                case ThreadSpawn::Linear:
                    return 1;
                case ThreadSpawn::Exp:
                    // Exp creates 1 thread at the first spawn step
                    const auto target = std::min(kMaxNumWorkers, curNumWorkers * 2);
                    return std::max(1UL, target - curNumWorkers);
            }
        }();
        DCHECK_LE(curNumWorkers + newWorkerCount, kMaxNumWorkers);
        mWorkers.reserve(kMaxNumWorkers);
        for (size_t i = 0; i < newWorkerCount; i++) {
            mWorkers.emplace_back(&ThreadPool::threadWorker, this);
        }
    }

    void threadWorker() {
        while (1) {
            // Wait until it has received a task to perform or a terminate signal
            std::unique_lock lock(mTaskMutex);
            mCvNewTask.wait(lock, [this] { return hasTodoTask() || mTerminateSignal; });

            if (mTerminateSignal) {
                return;
            }

            DCHECK(hasTodoTask());
            auto threadTask = mTaskBacklog.front();

            // Moving task from backlog to active workers count need to be a single atomic operation
            mTaskBacklog.pop_front();
            ++mNumActiveWorkers;

            // Successfully acquired the task to run, so it can release the mutex
            lock.unlock();

            // Run task
            threadTask();

            // Mark idle after task has finished running
            lock.lock();
            --mNumActiveWorkers;
            lock.unlock();

            mCvActiveWorkers.notify_all();
        }
    }

private:
    const size_t kMaxNumWorkers;
    std::vector<std::thread> mWorkers;
    size_t mNumActiveWorkers = 0;

    const ThreadSpawn kSpawnPolicy;

    // Task backlog queue
    std::deque<std::function<void()>> mTaskBacklog;

    bool mTerminateSignal = false;

    // A thread needs to acquire this mutex before accessing the task backlog
    std::mutex mTaskMutex;

    // Users of the thread pool must acquire this mutex to perform certain mutable operations
    std::mutex mUserMutex;

    std::condition_variable mCvNewTask;
    std::condition_variable mCvActiveWorkers;
};
