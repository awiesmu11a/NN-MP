
#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/MultiArrayDimension.h>
#include <ackermann_msgs/AckermannDriveStamped.h>

#include "tensorNet.h"
#include "FCN.h"

FCN* net = NULL;
float32_t* input = NULL;
float32_t* output = NULL;
std::vector<float32> feature_vector;
float acc = 0;
float steering_angle = 0;

cudaMalloc((void**)&input, 576 * sizeof(float32_t));

void features_callback(const std_msgs::Float32MultiArray::ConstPtr& msg)
{
    feature_vector = msg->data;
    input = feature_vector.data();

    if ( feature_vector.size() != 576 )
    {
        return;
    }

    cudaMemcpy(net.mInputs[0].CUDA, input, 576 * sizeof(float32_t), cudaMemcpyDeviceToDevice);

    if ( !net->Process() )
    {
        ROS_ERROR("Image feature extraction failed");
        return;
    }

    output = net.mOutputs[0].CPU;
    acc = output[0];
    steering_angle = output[1];

    return;
}

void control_callback(const ackermann_msgs::AckermannDriveStamped::ConstPtr& msg)
{
    ackermann_msgs::AckermannDriveStamped control_msg;
    control_msg.drive.speed = msg->drive.speed + 0.1 * acc;
    control_msg.drive.steering_angle = msg->drive.steering_angle + 0.1 * steering_angle;

    control_node.publish(control_msg);

    return;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "Actor");
    ros::NodeHandle n;

    std::string network = "actor_network";
    std::string prototxt_path = "";
    std::string model_path = "/home/nvidia/NN-ML/networks/models/actor.onnx";

    std::string input_blob = FCN_NET_DEFAULT_INPUT;
    std::string output_blob = FCN_NET_DEFAULT_OUTPUT;
    const Dims& features = Dims2(576, 1);

    net = FCN::Create(prototxt_path, model_path, input_blob, features, output_blob, maxBatchSize = 1);

    std::string features_sub = "ExtractedFeatures";
    std::string control = "low_level/ackermann_cmd_mux/input/teleop";
    std::string control_sub = "low_level/ackermann_cmd_mux/output";

    ros::Subscriber features_sub_node = n.subscribe(features_sub, 4, features_callback);
    ros::Subscriber control_sub_node = n.subscribe(control_sub, 4, control_callback);

    ros::Publisher control_node = n.advertise<ackermann_msgs::AckermannDriveStamped>(control, 4);

    ros::spin();

    // free resources from CUDA
}