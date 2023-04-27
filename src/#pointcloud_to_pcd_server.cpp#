#include "jsk_pcl_ros_utils/pointcloud_to_pcd.h"
#include <sensor_msgs/PointCloud2.h>
#include "jsk_recognition_utils/pcl_conversion_util.h"
#include <jsk_recognition_utils/tf_listener_singleton.h>
#include "wrapping/Convert.h"


#include <pcl/io/io.h>
#include <Eigen/Geometry>

class Server {
public:
  Server();
  ~Server() {};
  // void Server::convertServer(wrapping::Convert::Request &req,
  //                            wrapping::Convert::Response &res);
  void Server::convertServer();
  void loop();
};

Server::Server() {
  srv_ = boost::make_shared <dynamic_reconfigure::Server<Config> > (*pnh_);
  dynamic_reconfigure::Server<Config>::CallbackType f = boost::bind(&PointCloudToPCD::configCallback, this, _1, _2);
  srv_->setCallback (f);
  tf_listener_ = jsk_recognition_utils::TfListenerSingleton::getInstance();
  if(binary_)
    {
      if(compressed_)
        {
          ROS_INFO_STREAM ("Saving as binary compressed PCD");
        }
      else
        {
          ROS_INFO_STREAM ("Saving as binary PCD");
        }
    }
  else
    {
      ROS_INFO_STREAM ("Saving as binary PCD");
    }
}
// void Server::convertServer (wrapping::Convert::Request &req,
//                             wrapping::Convert::Response &res)
void Server::convertServer ()
  {
    pcl::PCLPointCloud2::ConstPtr cloud;
    cloud = ros::topic::waitForMessage<pcl::PCLPointCloud2>("input", *pnh_);
    if ((cloud->width * cloud->height) == 0)
    {
      return;
    }

    ROS_INFO ("Received %d data points in frame %s with the following fields: %s",
            (int)cloud->width * cloud->height,
            cloud->header.frame_id.c_str (),
            pcl::getFieldsList (*cloud).c_str ());

    Eigen::Vector4f v = Eigen::Vector4f::Zero ();
    Eigen::Quaternionf q = Eigen::Quaternionf::Identity ();
    if (!fixed_frame_.empty ()) {
        if (!tf_listener_->waitForTransform (fixed_frame_, cloud->header.frame_id, pcl_conversions::fromPCL (cloud->header).stamp, ros::Duration (duration_))) {
        ROS_WARN("Could not get transform!");
        return;
      }
      tf::StampedTransform transform_stamped;
      tf_listener_->lookupTransform (fixed_frame_, cloud->header.frame_id,  pcl_conversions::fromPCL (cloud->header).stamp, transform_stamped);
      Eigen::Affine3d transform;
      tf::transformTFToEigen(transform_stamped, transform);
      v = Eigen::Vector4f::Zero ();
      v.head<3> () = transform.translation ().cast<float> ();
      q = transform.rotation ().cast<float> ();
    }

    std::stringstream ss;
    ss << prefix_ << cloud->header.stamp << ".pcd";
    ROS_INFO ("Data saved to %s", ss.str ().c_str ());

    pcl::PCDWriter writer;
    if(binary_)
    {
      if(compressed_)
      {
        writer.writeBinaryCompressed (ss.str (), *cloud, v, q);
      }
      else
      {
        writer.writeBinary (ss.str (), *cloud, v, q);
      }
    }
    else
    {
      writer.writeASCII (ss.str (), *cloud, v, q, 8);
    }
  }

void Server::loop() {
  ros::Rate loop_rate(30);

  while(ros::ok()) {
    ros::spinOnce();
    loop_rate.sleep();
  }
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "point_clound_to_pcd_server");

  Server svr;
  svr.loop();
  return 0;
}
