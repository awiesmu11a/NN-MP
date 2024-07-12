
#include <queue>

#include <jetson-utils/cudaMappedMemory.h>

#include <ros/ros.h>
#include <ros/console.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/MultiArrayDimension.h>
#include <sensor_msgs/Image.h>

#include "tensorNet.h"
#include "imageConverter.h"
#include "CNN.h"

CNN* net = NULL;
imageConverter* input_cvt = NULL;

ros::Publisher img_feat_pub;

bool first_image = true;

size_t offset = 0;

std::queue<float*> input_queue;
float* output_features = NULL;

float* get_output( std::queue<float*> &input )
{

    std::queue<float*> temp_input = input;

    for (int i = 0; i < 4; i++)
    {
        float* temp = new float[64 * 64];
        std::memcpy(temp, temp_input.front(), 64 * 64 * sizeof(float));

        if ( !temp_input.empty() ) temp_input.pop();

        offset = i * 64 * 64 * sizeof(float);
        
        cudaMemcpy(net->GetInputPtr(0) + offset, temp, 64 * 64 * sizeof(float), cudaMemcpyHostToDevice);
        
        delete[] temp;
    }

    if ( !net->Process() )
    {
        ROS_ERROR("Image feature extraction failed");
        return 0;
    }

    return net->GetOutputPtr(0);
}

void depth_feat_extrac(const sensor_msgs::ImageConstPtr msg)
{

    if ( !input_cvt->assign(msg) && first_image )
    {
        ROS_ERROR("Allocation of memory failed");
        first_image = false;
        return;
    }

    if ( !input_cvt->Convert(msg) )
    {
        ROS_ERROR("Image conversion failed");
        return;
    }

    if ( input_queue.size() < 4 )
    {
        input_queue.push(input_cvt->GetOutputCPU());
        return;
    }
    else
    {
        input_queue.pop();
        input_queue.push(input_cvt->GetOutputCPU());
    }

    output_features = new float[512];

    CUDA(cudaMemcpy(output_features, get_output(input_queue), 512 * sizeof(float), cudaMemcpyDeviceToHost));

    std_msgs::Float32MultiArray img;
    img.layout.dim.push_back(std_msgs::MultiArrayDimension());
    img.layout.dim[0].size = 512;
    img.layout.dim[0].stride = 1;
    img.layout.dim[0].label = "img_feat";

    for (int i = 0; i < 512; i++) img.data.push_back(output_features[i]);

    img_feat_pub.publish(img);
    delete[] output_features;

    return;

}

int main(int argc, char **argv)
{
    const char* prototxt_path = "";
    const char* model_path = "/home/nvidia/NN-MP/networks/models/CNN.onnx";

    net = CNN::Create(prototxt_path, model_path);

    input_cvt = new imageConverter( 64, 64 );

    std::string cam_sub = "/camera/depth/image_rect_raw";
    std::string img_feat = "/image_features";

    ros::init(argc, argv, "ImageFeatureExtractor");
    ros::NodeHandle nh;

    ros::Subscriber img_sub = nh.subscribe(cam_sub, 4, depth_feat_extrac);
    img_feat_pub = nh.advertise<std_msgs::Float32MultiArray>(img_feat, 1);

    ros::spin();

    return 0;
}
