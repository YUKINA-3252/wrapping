#include "ros/ros.h"
#include "wrapping/Convert.h"

int main(int argc, char **afgv) {
  ros::init(argc, argv, "point_cloud_to_pcd_client");
  ros::NodeHandle nh;
  ros::ServiceClient client = nh.serviceClient<wrapping::Convert>("convert");

  wrapping::Convert srv;

  if (client.call(srv)) {
      ROS_INFO("Success");
    } else {
      ROS_ERROR("Failed");
      return 1;
    }
    return 0:
}
