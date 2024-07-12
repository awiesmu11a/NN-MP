
#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/MultiArrayDimension.h>
#include <ackermann_msgs/AckermannDriveStamped.h>

#include "tensorNet.h"
#include "FCN.h"

FCN* net = NULL;
float* input = NULL;
float* output = NULL;
std::vector<float> feature_vector;
float acc = 0;
float steering_angle = 0;
float current_speed = 0;
float current_steer = 0;

ros::Publisher control_node;

void features_callback(const std_msgs::Float32MultiArray::ConstPtr& msg)
{
    feature_vector = msg->data;
    input = new float[576];

    if ( feature_vector.size() != 576 ) return;

    std::memcpy(input, feature_vector.data(), 576 * sizeof(float));

    cudaMemcpy(net->GetInputPtr(0), input, 576 * sizeof(float), cudaMemcpyHostToDevice);

    if ( !net->Process() )
    {
        ROS_ERROR("Image feature extraction failed");
        return;
    }

    output = net->GetOutputPtr(0);
    acc = output[1]/5;
    steering_angle = output[0];

    std::cout<<"Acc "<<acc<<" steering angle"<<steering_angle<<std::endl;

    ackermann_msgs::AckermannDriveStamped control_msg;

    control_msg.drive.speed = current_speed + (0.05 * acc);
    if (current_speed >= 0.5) control_msg.drive.speed = 0.5;
    control_msg.drive.steering_angle = steering_angle;

    std::cout<<"Speed "<<control_msg.drive.speed<<" Steering angle"<<control_msg.drive.steering_angle<<std::endl;
    control_node.publish(control_msg);

    return;
}

void control_callback(const ackermann_msgs::AckermannDriveStamped::ConstPtr& msg)
{
    current_speed = msg->drive.speed;
    current_steer = msg->drive.steering_angle;

    return;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "Actor");
    ros::NodeHandle nh;

    std::string network = "actor_network";
    const char* prototxt_path = "";
    const char* model_path = "/home/nvidia/NN-MP/networks/models/actor.onnx";

    const Dims3& features = Dims3(1, 576, 1);

    net = FCN::Create(prototxt_path, model_path, features);

    std::string features_sub = "/ExtractedFeatures";
    std::string control = "/vesc/low_level/ackermann_cmd_mux/input/teleop";
    std::string control_sub = "/vesc/low_level/ackermann_cmd_mux/output";

    ros::Subscriber features_sub_node = nh.subscribe(features_sub, 4, features_callback);
    ros::Subscriber control_sub_node = nh.subscribe(control_sub, 4, control_callback);

    control_node = nh.advertise<ackermann_msgs::AckermannDriveStamped>(control, 4);

    ros::spin();

    return 0;
}
