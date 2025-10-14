#include "composition/listener_component.hpp"
#include <iostream>
#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include <event_camera_codecs/decoder.h>
#include <event_camera_codecs/decoder_factory.h>
using event_camera_codecs::EventPacket;

namespace composition
{

// Create a Listener "component" that subclasses the generic rclcpp::Node base class.
// Components get built into shared libraries and as such do not write their own main functions.
// The process using the component's shared library will instantiate the class as a ROS node.
Listener::Listener(const rclcpp::NodeOptions & options)
: Node("listener", options)
{
  polarity_buffer_.resize(1280 * 720, 0);
  // Create a callback function for when messages are received.
  // Variations of this function also exist using, for example, UniquePtr for zero-copy transport.
  auto callback =
    [this](const event_camera_codecs::EventPacketConstSharedPtr & msg) -> void
    {
      auto decoder = decoderFactory.getInstance(*msg);
      if (decoder) {
        decoder->decode(*msg, &processor);
      }
      RCLCPP_INFO(this->get_logger(), "Event counter: %d", count_);
      std::flush(std::cout);
    };

  // to get callbacks into MyProcessor, feed the message buffer
  // into the decoder like so
  /*
  void eventMsg(const event_camera_codecs::EventPacketConstSharedPtr & msg) {
    // will create a new decoder on first call, from then on returns existing one
    auto decoder = decoderFactory.getInstance(*msg);
    if (!decoder) { // msg->encoding was invalid
      return;
    }
    // the decode() will trigger callbacks to processor
    decoder->decode(*msg, &processor);
  }
    */

  /* To synchronize with frame based sensors it is useful to play back
    until a frame boundary is reached. The interface decodeUntil() is provided
    for this purpose. Do *NOT* use decode() and decodeUntil() on the same decoder!
    In the sample code belowframeTimes is an ordered queue of frame times.
    */

    /*
  void eventMsg2(const event_camera_codecs::EventPacketConstSharedPtr & msg) {
    auto decoder = decoderFactory.getInstance(*msg);
    uint64_t nextTime{0};
    // The loop will exit when all events in msg have been processed
    // or there are no more frame times available
    decoder->setTimeBase(msg->time_base);
    while (!frameTimes.empty() &&
      decoder->decodeUntil(*msg, &processor, frameTimes.front(), &nextTime)) {
      // use loop in case multiple frames fit inbetween two events
      while (!frameTimes.empty() && frameTimes.front() <= nextTime) {
        // processFrameHere()
        frameTimes.pop();
      }
    }
  }
  */

  // Create a subscription to the "/event_camera/events" topic which can be matched with one or more
  // compatible ROS publishers.
  // Note that not all publishers on the same topic with the same type will be compatible:
  // they must have compatible Quality of Service policies.
  auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).best_effort();
  sub_ = create_subscription<event_camera_codecs::EventPacket>("/event_camera/events", qos, callback);
  pub_ = create_publisher<sensor_msgs::msg::Image>("event_camera/filtered_image", 10);
  // Create timer for 20 FPS (50ms)
  timer_ = create_wall_timer(std::chrono::milliseconds(50), std::bind(&Listener::timer_callback, this));
}

void Listener::timer_callback()
{
  // Initialize Image message (1280x720, BGR8)
  sensor_msgs::msg::Image msg;
  msg.header.stamp = this->now();
  msg.header.frame_id = "1701";
  msg.height = 720;
  msg.width = 1280;
  msg.encoding = "bgr8";
  msg.is_bigendian = 0;
  msg.step = 1280 * 3; // 3 bytes per pixel (BGR)
  msg.data.resize(720 * 1280 * 3, 0); // Black image
  
  // Plot filtered events
  {
    std::lock_guard<std::mutex> lock(event_mutex_);
    for (const auto &i : pixel_indices_) {
      size_t idx = i * 3;
      msg.data[idx] = (polarity_buffer_[i] == 0) ? 255 : 0;
      msg.data[idx + 1] = 0;
      msg.data[idx + 2] = (polarity_buffer_[i] == 0) ? 0 : 255;
      polarity_buffer_[i] = 0;
    }
    pixel_indices_.clear();
  }

  // Publish image
  pub_->publish(msg);
}

}// namespace composition

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(composition::Listener)

