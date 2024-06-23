/*
Copyright 2022 Arsène Pérard-Gayot

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Source: https://github.com/madmann91/bvh/blob/master/src/bvh/v2/thread_pool.h
*/

#ifndef BVH_V2_THREAD_POOL_H
#define BVH_V2_THREAD_POOL_H

#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <queue>
#include <functional>

namespace bvh::v2 {

class ThreadPool {
public:
    using Task = std::function<void(size_t)>;

    /// Creates a thread pool with the given number of threads (a value of 0 tries to autodetect
    /// the number of threads and uses that as a thread count).
    ThreadPool(size_t thread_count = 0) { start(thread_count); }

    ~ThreadPool() {
        wait();
        stop();
        join();
    }

    inline void push(Task&& fun);
    inline void wait();

    size_t get_thread_count() const { return threads_.size(); }

private:
    static inline void worker(ThreadPool*, size_t);

    inline void start(size_t);
    inline void stop();
    inline void join();

    int busy_count_ = 0;
    bool should_stop_ = false;
    std::mutex mutex_;
    std::vector<std::thread> threads_;
    std::condition_variable avail_;
    std::condition_variable done_;
    std::queue<Task> tasks_;
};

void ThreadPool::push(Task&& task) {
    {
        std::unique_lock<std::mutex> lock(mutex_);
        tasks_.emplace(std::move(task));
    }
    avail_.notify_one();
}

void ThreadPool::wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [this] { return busy_count_ == 0 && tasks_.empty(); });
}

void ThreadPool::worker(ThreadPool* pool, size_t thread_id) {
    while (true) {
        Task task;
        {
            std::unique_lock<std::mutex> lock(pool->mutex_);
            pool->avail_.wait(lock, [pool] { return pool->should_stop_ || !pool->tasks_.empty(); });
            if (pool->should_stop_ && pool->tasks_.empty())
                break;
            task = std::move(pool->tasks_.front());
            pool->tasks_.pop();
            pool->busy_count_++;
        }
        task(thread_id);
        {
            std::unique_lock<std::mutex> lock(pool->mutex_);
            pool->busy_count_--;
        }
        pool->done_.notify_one();
    }
}

void ThreadPool::start(size_t thread_count) {
    if (thread_count == 0)
        thread_count = std::max(1u, std::thread::hardware_concurrency());
    for (size_t i = 0; i < thread_count; ++i)
        threads_.emplace_back(worker, this, i);
}

void ThreadPool::stop() {
    {
        std::unique_lock<std::mutex> lock(mutex_);
        should_stop_ = true;
    }
    avail_.notify_all();
}

void ThreadPool::join() {
    for (auto& thread : threads_)
        thread.join();
}

} // namespace bvh::v2

#endif
