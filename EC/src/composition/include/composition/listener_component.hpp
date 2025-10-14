#ifndef COMPOSITION__LISTENER_COMPONENT_HPP_
#define COMPOSITION__LISTENER_COMPONENT_HPP_

#include "composition/visibility_control.h"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "event_camera_codecs/decoder.h"
#include "event_camera_codecs/decoder_factory.h"
#include "sensor_msgs/msg/image.hpp"
#include <queue>
#include <deque>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>
#include <mutex>
#include <vector>

using event_camera_codecs::EventPacket;

namespace composition
{

class Listener : public rclcpp::Node
{
public:
  COMPOSITION_PUBLIC
  explicit Listener(const rclcpp::NodeOptions & options);

private:

  class MyProcessor : public event_camera_codecs::EventProcessor
  {
  public:
    explicit MyProcessor(Listener * parent) : parent_(parent) {
      event_counts_.resize(1280 * 720, 0);
      event_times_.resize(1280 * 720, 0);
    }
    inline void eventCD(uint64_t t, uint16_t ex, uint16_t ey, uint8_t polarity) override {

      size_t idx = ey * 1280 + ex;
      
      // Define box parameters
      const int delta_x = 1;  // ±x pixels
      const int delta_y = 1;  // ±y pixels
      const uint64_t delta_t = 50000000;  // 50ms in nanoseconds to the past

      // Update event time and count
      if (event_times_[idx] == 0 || (t - event_times_[idx]) <= delta_t) {
        event_times_[idx] = t;
        event_counts_[idx]++;
      } else {
        event_counts_[idx] = 1;
        event_times_[idx] = t;
      }

      // Check neighboring pixels
      int min_events = 4;
      int event_count = 0;
      for (int dy = -delta_y; dy <= delta_y; ++dy) {
        for (int dx = -delta_x; dx <= delta_x; ++dx) {
          int nx = ex + dx;
          int ny = ey + dy;
          if (nx >= 0 && nx < 1280 && ny >= 0 && ny < 720) {
            event_count += event_counts_[ny * 1280 + nx];
          }
          if (event_count > min_events) {
            break; // Stop checking if more than 4 events found
          }
        }
        if (event_count > min_events) {
          break; // Stop checking if more than 4 events found
        }
      }

      // Keep event if more than min_events events in box
      if (event_count > min_events) {
        std::lock_guard<std::mutex> lock(parent_->event_mutex_);
        parent_->count_++;
        parent_->pixel_indices_.push_back(idx);
        parent_->polarity_buffer_[idx] = polarity; 
      }

      // Track active indices
      active_indices_.push_back(idx);

      // Periodic cleanup
      while (!active_indices_.empty() && (t - event_times_[active_indices_.front()]) > delta_t) {
        size_t i = active_indices_.front();
        event_times_[i] = 0;
        event_counts_[i] = 0;
        active_indices_.pop_front();
      }

    }
    void eventExtTrigger(uint64_t, uint8_t, uint8_t) override {}
    // called after no more events decoded in this packet
    void finished() override{};
    void rawData(const char *, size_t) override{};  // passthrough of raw data
  private:
    Listener * parent_;
    std::vector<uint64_t> event_times_;
    std::vector<int> event_counts_;
    std::deque<size_t> active_indices_; 
  };
  
  COMPOSITION_PUBLIC
  void timer_callback();
  rclcpp::Subscription<event_camera_codecs::EventPacket>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::queue<uint64_t> frameTimes; // For frame synchronization
  int count_ = 0;
  std::mutex event_mutex_;
  std::vector<size_t> pixel_indices_;
  std::vector<uint8_t> polarity_buffer_;
  MyProcessor processor{this};
  event_camera_codecs::DecoderFactory<EventPacket, MyProcessor> decoderFactory;
};  

} // namespace composition

#endif  // COMPOSITION__LISTENER_COMPONENT_HPP_