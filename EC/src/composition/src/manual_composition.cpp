#include <memory>
#include "composition/listener_component.hpp"
#include "composition/talker_component.hpp"
#include "rclcpp/rclcpp.hpp"

int main(int argc, char * argv[])
{
  setvbuf(stdout, NULL, _IONBF, BUFSIZ);
  rclcpp::init(argc, argv);

  // Create an executor that will be responsible for execution of callbacks for a set of nodes.
  // With this version, all callbacks will be called from within this thread (the main one).
  rclcpp::executors::SingleThreadedExecutor exec;
  rclcpp::NodeOptions options;

  // Add some nodes to the executor which provide work for the executor during its "spin" function.
  // auto talker = std::make_shared<composition::Talker>(options);
  // exec.add_node(talker);
  auto listener = std::make_shared<composition::Listener>(options);
  exec.add_node(listener);

  exec.spin();
  rclcpp::shutdown();
  return 0;
}