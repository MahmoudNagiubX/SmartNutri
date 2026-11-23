#pragma once

#include <stddef.h>

namespace smartnutri {

template <size_t Capacity>
class MedianEmaFilter {
 public:
  explicit MedianEmaFilter(float alpha = 0.35f)
      : count_(0), next_(0), alpha_(alpha), ema_(0.0f), has_ema_(false) {}

  float update(float input) {
    values_[next_] = input;
    next_ = (next_ + 1) % Capacity;
    if (count_ < Capacity) {
      ++count_;
    }

    float sorted[Capacity] = {};
    for (size_t i = 0; i < count_; ++i) {
      sorted[i] = values_[i];
    }
    for (size_t i = 1; i < count_; ++i) {
      const float value = sorted[i];
      size_t j = i;
      while (j > 0 && sorted[j - 1] > value) {
        sorted[j] = sorted[j - 1];
        --j;
      }
      sorted[j] = value;
    }

    const float median = count_ % 2 == 0
                             ? (sorted[count_ / 2 - 1] + sorted[count_ / 2]) * 0.5f
                             : sorted[count_ / 2];
    if (!has_ema_) {
      ema_ = median;
      has_ema_ = true;
    } else {
      ema_ += alpha_ * (median - ema_);
    }
    return ema_;
  }

  void reset() {
    count_ = 0;
    next_ = 0;
    ema_ = 0.0f;
    has_ema_ = false;
  }

 private:
  float values_[Capacity] = {};
  size_t count_;
  size_t next_;
  float alpha_;
  float ema_;
  bool has_ema_;
};

template <size_t Capacity>
class StabilityWindow {
 public:
  StabilityWindow() : count_(0), next_(0) {}

  void add(float value) {
    values_[next_] = value;
    next_ = (next_ + 1) % Capacity;
    if (count_ < Capacity) {
      ++count_;
    }
  }

  void reset() {
    count_ = 0;
    next_ = 0;
  }

  size_t sampleCount() const { return count_; }

  bool range(float& result) const {
    if (count_ == 0) {
      return false;
    }
    float minimum = values_[0];
    float maximum = values_[0];
    for (size_t i = 1; i < count_; ++i) {
      if (values_[i] < minimum) minimum = values_[i];
      if (values_[i] > maximum) maximum = values_[i];
    }
    result = maximum - minimum;
    return true;
  }

  bool isStable(float threshold) const {
    float current_range = 0.0f;
    return count_ == Capacity && range(current_range) && current_range <= threshold;
  }

 private:
  float values_[Capacity] = {};
  size_t count_;
  size_t next_;
};

inline float weightDeltaGrams(float current_g, float previous_g) {
  return current_g - previous_g;
}

}  // namespace smartnutri
