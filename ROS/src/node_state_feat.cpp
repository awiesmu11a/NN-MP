
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <queue>

#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/MultiArrayDimension.h>
#include <ackermann_msgs/AckermannDriveStamped.h>
#include <nav_msgs/Odometry.h>

#include "tensorNet.h"
#include "FCN.h"

FCN* net = NULL;

std::vector<float32> rel_goal;
stc::vector<float32> image_features;
std::vector<float32> states;

std::queue<float32> car_states;
std::queue<float32> temp;

std::vector<float32> goal;

float32 x, y, z;
float32 q1, q2, q3, q4;
float32 roll, pitch, yaw;

float32 velocity, steering_angle;

Eigen::Matrix2d rot_mat;
Eigen::Vector2d rel_goal_vec;

void imag_feat_extrac( const std_msgs::Float32MultiArray::ConstPtr& msg )
{
    image_features = msg->data;
    return;

}

void state_feat_extrac( const ackermann::AckermannDriveStamped::ConstPtr& msg )
{
    if ( image_features.size() != 512 )
    {
        return;
    }
    states = image_features;

    velocity = msg->drive.speed;
    steering_angle = msg->drive.steering_angle;

    if ( car_states.size() < 12) {
        car_states.push(rel_goal[0]);
        car_states.push(rel_goal[1]);
        car_states.push(velocity);
        car_states.push(steering_angle);
        return;
    }
    else {
        car_states.pop();
        car_states.pop();
        car_states.pop();
        car_states.pop();
        car_states.push(rel_goal[0]);
        car_states.push(rel_goal[1]);
        car_states.push(velocity);
        car_states.push(steering_angle);
    }

    cudaMemcpy(net.mInputs[0].CUDA, car_states, 12 * sizeof(float32), cudaMemcpyHostToDevice);

    if ( !net->Process() )
    {
        ROS_ERROR("State feature extraction failed");
        return;
    }

    for (int i = 0; i < 64; i++)
    {
        states.push_back(net.mOutputs[0].CPU[i]);
    }

    std_msgs::Float32MultiArray feat;
    feat.layout.dim.push_back(std_msgs::MultiArrayDimension());
    feat.layout.dim[0].size = 576;
    feat.layout.dim[0].stride = 1;
    feat.layout.dim[0].label = "states";
    for (int i = 0; i < 576; i++)
    {
        feat.data.push_back(states[i]);
    }

    feat_pub.publish(feat);

    return;


}

void odom_feat_extrac( const nav_msgs::Odometry::ConstPtr& msg )
{
    x = msg->pose.pose.position.x;
    y = msg->pose.pose.position.y;
    z = msg->pose.pose.position.z;
    
    q1 = msg->pose.pose.orientation.x;
    q2 = msg->pose.pose.orientation.y;
    q3 = msg->pose.pose.orientation.z;
    q4 = msg->pose.pose.orientation.w;

    rel_goal[0] = goal[0] - x;
    rel_goal[1] = goal[1] - y;

    rel_goal_vec[0] = rel_goal[0];
    rel_goal_vec[1] = rel_goal[1];

    float32 roll = atan2(2 * (q4 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2));
    float32 pitch = asin(2 * (q4 * q2 - q3 * q1));
    float32 yaw = atan2(2 * (q4 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3));

    rot_mat[0][0] = cos(yaw);
    rot_mat[0][1] = sin(yaw);
    rot_mat[1][0] = -sin(yaw);
    rot_mat[1][1] = cos(yaw);

    rel_goal_vec = rot_mat * rel_goal_vec;

    rel_goal[0] = rel_goal_vec[0];
    rel_goal[1] = rel_goal_vec[1];

    return;
}

int main(int argc, char **argv)
{
    goal.push_back(0.0);
    goal.push_back(0.0);

    rel_goal = goal;

    ros::init(argc, argv, "FeatureExtractor");
    ros::NodeHandle nh;

    std::string network = "State_Feat_Extractor";
    std::string prototxt_path = "";
    std::string model_path = "/home/nvidia/NN-ML/networks/models/state_fcn.onnx";

    std::string input_blob_state = FCN_NET_DEFAULT_INPUT;
    std::string output_blob_state = FCN_NET_DEFAULT_OUTPUT;
    const Dims2& states_input_dim = Dims2(12, 1);

    net = FCN::Create(prototxt_path, model_path, input_blob_state, states_input_dim, output_blob_state, maxBatchSize = 1);

    std::string odom_sub = "/vesc/odom";
    std::string cmd_sub = "low_level/ackermann_cmd_mux/output"
    std::string features = "/ExtractedFeatures";
    std::string img_feat = "/image_features";

    ros::Subscriber odom_sub_node = nh.subscribe(odom_sub, 4, odom_feat_extrac);
    ros::Subscriber cmd_sub_node = nh.subscribe(cmd_sub, 4, state_feat_extrac);
    ros::Subscriber img_sub_node = nh.subscribe(img_feat, 1, img_feat_extrac);

    ros::Publisher feat_pub_node = nh.advertise<std_msgs::Float32MultiArray>(features, 4);

    ros::spin();

    // free resources from CUDA
    
}