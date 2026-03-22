// semantic_map_main.cpp — 独立运行入口 (调试用)
#include "semantic_vslam/semantic_map_node.hpp"
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<semantic_vslam::SemanticMapNode>());
  rclcpp::shutdown();
  return 0;
}
