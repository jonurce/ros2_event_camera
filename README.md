# ROS2 Humble package for reading and filtering events from Event Camera Metavision® EVK4 – HD
This repository provides a package for ROS2 humble to read the topic "/event_camera/events", filter events using a spatio temporal boundary box, and publish the filtered event in image format (sensor_msgs/msg/image) to the topic "event_camera/filtered_image".

The packages inside src are distributed as follows:  

```   
src  
├── composition             # Node which reads, filters and publishes the event data, coded in c++ and using ROS2 components for high speed processing.
├── event_camera_codecs     # Package cloned from (https://github.com/ros-event-camera/event_camera_codecs).    
└── event_reader            # Package with python nodes to read event data, for testing purposes.    
```  




