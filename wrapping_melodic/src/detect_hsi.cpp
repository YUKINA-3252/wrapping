#include <cmath>
#include <ros/ros.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float32MultiArray.h>
#include <pcl/point_cloud.h>
#include <pcl/common/io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/common.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/segmentation/progressive_morphological_filter.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_listener.h>


double average_x = 0.0;
double average_y = 0.0;
double average_z = 0.0;
double x_coord_min = 0.0;
double x_coord_max = 0.0;
double y_coord_min = 0.0;
double y_coord_max = 0.0;
geometry_msgs::PointStamped target_point;
ros::Publisher pub, pub_pcl;


void pclCallback(const sensor_msgs::PointCloud2 msg){
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud_nan(new pcl::PointCloud<pcl::PointXYZRGB>);
  // pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
  // pcl::ExtractIndices<pcl::PointXYZ> extract;
  pcl::fromROSMsg(msg, *pcl_cloud_nan);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

  double sum_x = 0.0;
  double sum_y = 0.0;
  double sum_z = 0.0;

  tf2_ros::Buffer tfBuffer;
  tf2_ros::TransformListener listener(tfBuffer);

  std::string source_frame = "WAIST";
  std::string target_frame = "head_camera_rgb_optical_frame";
  geometry_msgs::TransformStamped transform;
  transform = tfBuffer.lookupTransform(source_frame, target_frame, ros::Time(), ros::Duration(3.0));

  int pcl_cloud_size = 0;
  for (int i = 0; i < (*pcl_cloud_nan).size(); i++) {
    if (!std::isnan(pcl_cloud_nan->points[i].x) && !std::isnan(pcl_cloud_nan->points[i].y) && !std::isnan(pcl_cloud_nan->points[i].z)) {
      geometry_msgs::PointStamped source_point;
      source_point.header.frame_id = source_frame;
      source_point.point.x = pcl_cloud_nan->points[i].x;
      source_point.point.y = pcl_cloud_nan->points[i].y;
      source_point.point.z = pcl_cloud_nan->points[i].z;
      tf2::doTransform(source_point, target_point, transform);
      if ((target_point.point.x > x_coord_min) && (target_point.point.x < x_coord_max) && (target_point.point.y > y_coord_min) && (target_point.point.y < y_coord_max)) {
        pcl_cloud_size ++;
        sum_x += target_point.point.x;
        sum_y += target_point.point.y;
        sum_z += target_point.point.z;

        filtered_cloud->points.push_back(pcl_cloud_nan->points[i]);
      }
    }
  }
  average_x = sum_x / pcl_cloud_size;
  average_y = sum_y / pcl_cloud_size;
  average_z = sum_z / pcl_cloud_size;

  // tf2_ros::Buffer tfBuffer;
  // tf2_ros::TransformListener listener(tfBuffer);

  // std::string source_frame = "WAIST";
  // std::string target_frame = "head_camera_rgb_optical_frame";
  // geometry_msgs::PointStamped source_point;
  // source_point.header.frame_id = source_frame;
  // source_point.point.x = average_x;
  // source_point.point.y = average_y;
  // source_point.point.z = average_z;

  // try {
  //   geometry_msgs::TransformStamped transform;
  //   transform = tfBuffer.lookupTransform(source_frame, target_frame, ros::Time(), ros::Duration(3.0));
  //   tf2::doTransform(source_point, target_point, transform);

  // } catch (tf2::TransformException& ex) {
  //   ROS_ERROR("failed to transform point: %s", ex.what());
  // }

  // publish
  std_msgs::Float32MultiArray average_msg;
  average_msg.data.push_back(average_x);
  average_msg.data.push_back(average_y);
  average_msg.data.push_back(average_z);

  sensor_msgs::PointCloud2 filtered_cloud_msg;
  pcl::toROSMsg(*filtered_cloud, filtered_cloud_msg);
  filtered_cloud_msg.header = msg.header;

  pub.publish(average_msg);
  pub_pcl.publish(filtered_cloud_msg);
}


int main(int argc, char** argv){
  ros::init(argc, argv, "pcl_listener_cpp");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  std::string sub_topic, pub_topic, pub_pcl_topic;
  pnh.getParam("extract_point_cloud_topic", sub_topic);
  pnh.getParam("average_coord_topic", pub_topic);
  pnh.getParam("x_coord_min", x_coord_min);
  pnh.getParam("x_coord_max", x_coord_max);
  pnh.getParam("y_coord_min", y_coord_min);
  pnh.getParam("y_coord_max", y_coord_max);
  pnh.getParam("publish_pcl_topic", pub_pcl_topic);

  ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>(sub_topic, 1, pclCallback);
  pub = nh.advertise<std_msgs::Float32MultiArray>(pub_topic, 10);
  pub_pcl = nh.advertise<sensor_msgs::PointCloud2>(pub_pcl_topic, 10);

  ros::spin();

  return 0;
}
