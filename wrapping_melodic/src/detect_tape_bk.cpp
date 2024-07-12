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

#define PI 3.14159265359

double average_x = 0.0;
double average_y = 0.0;
double average_z = 0.0;
double average_r = 0.0;
double average_g = 0.0;
double average_b = 0.0;
double average_h = 0.0;
double average_s = 0.0;
double average_i = 0.0;
float min_hsi_[3];
float max_hsi_[3];
geometry_msgs::PointStamped target_point;
ros::Publisher pub;
ros::Publisher pub_hsi;

void rgbToHsi(uint8_t R, uint8_t G, uint8_t B, float& h, float& s, float&i) {
  // float r = R / 255.0f;
  // float g = G / 255.0f;
  // float b = B / 255.0f;

  // float cMax = std::max(B, std::max(R, G));
  // float cMin = std::min(B, std::max(R, G));
  // float delta = cMax - cMin;

  // if (delta == 0) h = 0;
  // else if (cMax == r) h = (int)((60 * ((g - b) / delta) + 360)) % 360;
  // else if (cMax == g) h = (int)((60 * ((b - r) / delta) + 120)) % 360;
  // else h = (int)((60 * ((r - g) / delta) + 240)) % 360;

  // if (cMax == 0) s = 0;
  // else s = 1 - (cMin / (r + g + b) * 3);
  // // else s = delta / cMax;

  // i = (r + g + b) / 3;


  float r = R / (R + G + B);
  float g = G / (R + G + B);
  float b = B / (R + G + B);

  float cMax = std::max(b, std::max(r, g));
  float cMin = std::min(b, std::max(r, g));

  if (b <= g) h = acos((0.5 * ((r - g) + (r - b))) / pow((pow(r - g, 2) + (r - b) * (g - b)), 0.5));
  else 2 * PI - acos((0.5 * ((r - g) + (r - b))) / pow((pow(r - g, 2) + (r - b) * (g - b)), 0.5));


  s = 1 - 3 * cMin;
  i = (R + G + B) / (3 * 255);

  h = 255 / 360 * h - 128;
  s = 255 * s;
  i = 255 * i;

  // double M1 = sqrt(6) / 3 * r - sqrt(6) / r * g - sqrt(6) / 6 * b;
  // double M2 = sqrt(2) / 2 * g - sqrt(2) / 2 * b;
  // double I1 = -sqrt(6) / 6 * r - sqrt(2) / 2 * g + sqrt(3) / 3 * b;

  // h = atan(M1 / M2);
  // s = sqrt(pow(M1, 2) + pow(M2, 2));
  // i = sqrt(3) * I1;

  // float I_max = std::max(B, std::max(R, G));
  // float I_min = std::min(B, std::min(R, G));
  // float delta = I_max - I_min;

  // float r = (I_max - R) / delta;
  // float g = (I_max - G) / delta;
  // float b = (I_max - B) / delta;

  // i = I_max;
  // s = delta / I_max;
  // if (R == I_max && G == I_min) h = (5 + b) / 6;
  // else if (R == I_max && G != I_min) h = (1 - g) / 6;
  // else if (G == I_max && B == I_min) h = (1 + r) / 6;
  // else if (G == I_max && B != I_min) h = (3 - b) / 6;
  // else if (B == I_max && R == I_min) h = (3 + g) / 6;
  // else if (b == I_max && R != I_min) h = (5 - r) / 6;

  // i = (cMax + cMin) / 2.0f;

  // s = delta / cMax;

  // if (delta==0) {
  //   double XR = (cMax - r) / delta;
  //   double XG = (cMax - g) / delta;
  //   double XB = (cMax - b) / delta;

  //   if (XR == cMax) h = (XB - XG) * PI / 3;
  //   if (XG == cMax) h = (2 + r - g) * PI / 3;
  //   if (XB == cMax) h = (4 + g - r) * PI / 3;
  //   if (h < 0) h += 2 * PI;
  // } else {
  //   h = 0.0;
  // }
}


void pclCallback(const sensor_msgs::PointCloud2 msg){
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud_nan(new pcl::PointCloud<pcl::PointXYZRGB>);
  // pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
  // pcl::ExtractIndices<pcl::PointXYZ> extract;
  pcl::fromROSMsg(msg, *pcl_cloud_nan);

  double sum_x = 0.0;
  double sum_y = 0.0;
  double sum_z = 0.0;
  double sum_r = 0.0;
  double sum_g = 0.0;
  double sum_b= 0.0;
  double sum_h = 0.0;
  double sum_s = 0.0;
  double sum_i = 0.0;

  min_hsi_[0] = min_hsi_[1] = min_hsi_[2] = std::numeric_limits<float>::max();
  max_hsi_[0] = max_hsi_[1] = max_hsi_[2] = -std::numeric_limits<float>::max();

  int pcl_cloud_size = 0;
  int hsi_size = 0;
  for (int i = 0; i < (*pcl_cloud_nan).size(); i++) {
    if (!std::isnan(pcl_cloud_nan->points[i].x) && !std::isnan(pcl_cloud_nan->points[i].y) && !std::isnan(pcl_cloud_nan->points[i].z)) {
      pcl_cloud_size ++;
      sum_x += pcl_cloud_nan->points[i].x;
      sum_y += pcl_cloud_nan->points[i].y;
      sum_z += pcl_cloud_nan->points[i].z;

      uint8_t r = pcl_cloud_nan->points[i].r;
      uint8_t g = pcl_cloud_nan->points[i].g;
      uint8_t b = pcl_cloud_nan->points[i].b;
      if ((r > 200) && (g > 200)) {

          float h, s, i;
          rgbToHsi(r, g, b, h, s, i);

          // min_hsi_[0] = std::min(min_hsi_[0], h);
          // min_hsi_[1] = std::min(min_hsi_[1], s);
          // min_hsi_[2] = std::min(min_hsi_[2], i);

          // max_hsi_[0] = std::max(max_hsi_[0], h);
          // max_hsi_[1] = std::max(max_hsi_[1], s);
          // max_hsi_[2] = std::max(max_hsi_[2], i);

          sum_h += h;
          sum_s += s;
          sum_i += i;
          sum_r += r;
          sum_g += g;
          sum_b += b;

          hsi_size ++;
      }
    }
  }
  average_x = sum_x / pcl_cloud_size;
  average_y = sum_y / pcl_cloud_size;
  average_z = sum_z / pcl_cloud_size;
  average_r = sum_r / hsi_size;
  average_g = sum_g / hsi_size;
  average_b = sum_b / hsi_size;
  average_h = sum_h / hsi_size;
  average_s = sum_s / hsi_size;
  average_i = sum_i / hsi_size;

  tf2_ros::Buffer tfBuffer;
  tf2_ros::TransformListener listener(tfBuffer);

  std::string source_frame = "WAIST";
  std::string target_frame = "head_camera_rgb_optical_frame";
  geometry_msgs::PointStamped source_point;
  source_point.header.frame_id = source_frame;
  source_point.point.x = average_x;
  source_point.point.y = average_y;
  source_point.point.z = average_z;

  try {
    geometry_msgs::TransformStamped transform;
    transform = tfBuffer.lookupTransform(source_frame, target_frame, ros::Time(), ros::Duration(3.0));
    tf2::doTransform(source_point, target_point, transform);

  } catch (tf2::TransformException& ex) {
    ROS_ERROR("failed to transform point: %s", ex.what());
  }

  // publish
  std_msgs::Float32MultiArray average_msg;
  average_msg.data.push_back(target_point.point.x);
  average_msg.data.push_back(target_point.point.y);
  average_msg.data.push_back(target_point.point.z);

  std_msgs::Float32MultiArray hsi_msg;
  // hsi_msg.data.push_back(min_hsi_[0]);
  // hsi_msg.data.push_back(min_hsi_[1]);
  // hsi_msg.data.push_back(min_hsi_[2]);
  // hsi_msg.data.push_back(max_hsi_[0]);
  // hsi_msg.data.push_back(max_hsi_[1]);
  // hsi_msg.data.push_back(max_hsi_[2]);
  hsi_msg.data.push_back(average_h);
  hsi_msg.data.push_back(average_s);
  hsi_msg.data.push_back(average_i);
  hsi_msg.data.push_back(average_r);
  hsi_msg.data.push_back(average_g);
  hsi_msg.data.push_back(average_b);

  pub.publish(average_msg);
  pub_hsi.publish(hsi_msg);
}


int main(int argc, char** argv){
  ros::init(argc, argv, "pcl_listener_cpp");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  std::string sub_topic, pub_topic, pub_topic_hsi;
  pnh.getParam("extract_point_cloud_topic", sub_topic);
  pnh.getParam("average_point_cloud_topic", pub_topic);
  pnh.getParam("min_max_hsi_topic", pub_topic_hsi);

  ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>(sub_topic, 1, pclCallback);
  pub = nh.advertise<std_msgs::Float32MultiArray>(pub_topic, 10);
  pub_hsi = nh.advertise<std_msgs::Float32MultiArray>(pub_topic_hsi, 10);

  ros::spin();

  return 0;
}
