
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <queue>

#include <ros/ros.h>
#include <ros/console.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/MultiArrayDimension.h>
#include <ackermann_msgs/AckermannDriveStamped.h>
#include <nav_msgs/Odometry.h>

#include "tensorNet.h"
#include "FCN.h"

// Order of queue could be different

bool goal_reached = false;

FCN* net = NULL;

std::vector<float> rel_goal;
std::vector<float> image_features;
std::vector<float> states;

std::queue<float> car_states;

float* temp = NULL;
size_t state_buffer = 12;

std::vector<float> goal;

float x, y, z;
float q1, q2, q3, q4;
float roll, pitch, yaw;

float velocity, steering_angle;

Eigen::Matrix2d rot_mat;
Eigen::Vector2d rel_goal_vec;

ros::Publisher feat_pub_node;

void img_feat_extrac( const std_msgs::Float32MultiArray::ConstPtr& msg )
{
    image_features = msg->data;
    if (goal_reached) image_features.empty();
    return;

}

void state_feat_extrac( const ackermann_msgs::AckermannDriveStamped::ConstPtr& msg )
{
    if ( goal_reached ) 
    {
        std::cout<<"GOAL_REACHED"<<std::endl;
        states.empty();
        return;
    }

    if ( image_features.size() != 512 ) return;
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

    temp = new float[state_buffer];

    std::memcpy(temp, &car_states.front(), state_buffer * sizeof(float));

    cudaMemcpy(net->GetInputPtr(0), temp, 12 * sizeof(float), cudaMemcpyHostToDevice);

    if ( !net->Process() )
    {
        ROS_ERROR("State feature extraction failed");
        return;
    }

    for (int i = 0; i < 64; i++) states.push_back(net->GetOutputPtr(0)[i]);

    std_msgs::Float32MultiArray feat;
    feat.layout.dim.push_back(std_msgs::MultiArrayDimension());
    feat.layout.dim[0].size = 576;
    feat.layout.dim[0].stride = 1;
    feat.layout.dim[0].label = "states";
    for (int i = 0; i < 576; i++) feat.data.push_back(states[i]);

    feat_pub_node.publish(feat);

    return;
}

void odom_feat_extrac( const nav_msgs::Odometry::ConstPtr& msg )
{
    if(goal_reached)
    {
        std::cout<<"GOAL_REACHED"<<std::endl;
        return;
    }

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

    roll = atan2(2 * (q4 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2));
    pitch = asin(2 * (q4 * q2 - q3 * q1));
    yaw = atan2(2 * (q4 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3));

    rot_mat(0, 0) = cos(yaw);
    rot_mat(0, 1) = sin(yaw);
    rot_mat(1, 0) = sin(yaw);
    rot_mat(1, 1) = -cos(yaw);

    rel_goal_vec = rot_mat * rel_goal_vec;

    rel_goal[0] = rel_goal_vec[0];
    rel_goal[1] = rel_goal_vec[1];

    std::cout<<rel_goal[0]<<" "<<rel_goal[1]<<std::endl;


    if ( ((rel_goal[0] * rel_goal[0]) + (rel_goal[1] * rel_goal[1])) < 0.25 )
    {
        std::cout<<"Reached"<<std::endl;
        goal_reached = true;
    }

    return;
}

int main(int argc, char **argv)
{
    if (argc == 3) 
    {
        goal.push_back(std::stod(argv[1]));
        goal.push_back(std::stod(argv[2]));
    }
    else
    {
        goal.push_back(1.0);
        goal.push_back(0.0);
    }

    rel_goal = goal;

    ros::init(argc, argv, "FeatureExtractor");
    ros::NodeHandle nh;
    
    std::string network = "State_Feat_Extractor";
    const char* prototxt_path = "";
    const char* model_path = "/home/nvidia/NN-MP/networks/models/FCN.onnx";

    const Dims3& states_input_dim = Dims3(1, 12, 1);

    net = FCN::Create(prototxt_path, model_path, states_input_dim);

    std::string odom_sub = "/vesc/odom";
    std::string cmd_sub = "/vesc/low_level/ackermann_cmd_mux/output";
    std::string features = "/ExtractedFeatures";
    std::string img_feat = "/image_features";

    ros::Subscriber odom_sub_node = nh.subscribe(odom_sub, 4, odom_feat_extrac);
    ros::Subscriber cmd_sub_node = nh.subscribe(cmd_sub, 4, state_feat_extrac);
    ros::Subscriber img_sub_node = nh.subscribe(img_feat, 1, img_feat_extrac);

    feat_pub_node = nh.advertise<std_msgs::Float32MultiArray>(features, 1);

    ros::spin();

    return 0;
    
}