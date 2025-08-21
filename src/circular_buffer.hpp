#pragma once

#include <vector>
#include <spdlog/spdlog.h>

// not thread-safe
// Implements insert in O(1) time and contents in O(capacity) time with the invariant that
// the buffer will only hold the last capacity elemetns inserted.
template <typename T>
class CircularBuffer {
public:
  // Constructor: initializes a circular buffer with given max capacity
  CircularBuffer(size_t capacity): capacity_(capacity), size_(0), left_(0), right_(0) {
    buf_.resize(capacity);
  };

  // Destructor
  ~CircularBuffer() {};

  // Returns the current contents of the CircularBuffer starting from the left pointer,
  // ending in the right, cycling indices mod capacity.
  std::vector<T> contents() const {
    std::vector<T> result;
    result.reserve(size());
    if (empty()) return result;

    size_t current = left_;
    for (size_t i = 0; i < size(); i++) {
      result.push_back(buf_[current]);
      current = (current + 1) % capacity_;
    }

    return result;
  }

  void push(const T& x) {
    if (full()) {
      left_ = (left_ + 1) % capacity_;
    } else {
      size_++;
    }

    buf_[right_] = x;
    right_ = (right_ + 1) % capacity_;
  }

  void pop() {
    if (!empty()) {
      left_ = (left_ + 1) % capacity_;
      size_--;
    }
  }

  void reset() {
    left_ = 0;
    right_ = 0;
    size_ = 0;
  }

  size_t capacity() const { return capacity_; }
  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }
  bool full() const { return size_ == capacity_; }
protected:

private:
  std::vector<T> buf_;
  size_t capacity_;
  size_t size_;
  int left_;
  int right_;
}; // class CircularBuffer