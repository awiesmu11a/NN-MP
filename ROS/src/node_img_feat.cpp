
#include <std::queue>

#include <jetson-utils/cudaMappedMemory.h>

#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/MultiArrayDimension.h>
#include <sensor_msgs/Image.h>

#include "tensorNet.h"
#include "imageConverter.h"
#include "CNN.h"

CNN* net = NULL;
input_cvt = NULL;

float32_t* temp = NULL;
size_t offset = 0;

std::queue<float32_t*> input_queue;
float32_t* output_features = NULL;

float32_t* get_output( std::queue<float32_t*> &input )
{
    std::queue<float32_t*> temp_input = input;
    temp = NULL;

    for (int i = 0; i < 4; i++)
    {
        temp = temp_input.pop();
        offset = i * 64 * 64 * sizeof(float32_t);
        cudaMemcpy(net->GetInputPtr(0) + offset, temp, 64 * 64 * sizeof(float32_t), cudaMemcpyDeviceToDevice);
        free(temp);
    }

    if ( !net->Process() )
    {
        ROS_ERROR("Image feature extraction failed");
        return;
    }

    delete temp_input;

    return net->GetOutputPtr(0);
}

void depth_feat_extrac(const sensor_msgs::ImageConstPtr msg)
{
    if ( !input_cvt->Convert(msg) )
    {
        ROS_ERROR("Image conversion failed");
        return;
    }

    if ( input_queue.size() < 4 )
    {
        input_queue.push(input_cvt->GetOutputGPU());
        return;
    }
    else
    {
        input_queue.pop();
        input_queue.push(input_cvt->GetOutputGPU());
    }

    output_features = get_output(input_queue);

    std_msgs::Float32MultiArray img_feat;
    img_feat.layout.dim.push_back(std_msgs::MultiArrayDimension());
    img_feat.layout.dim[0].size = 512;
    img_feat.layout.dim[0].stride = 1;
    img_feat.layout.dim[0].label = "img_feat";

    for (int i = 0; i < 512; i++)
    {
        img_feat.data.push_back(output_features[i]);
    }

    img_feat_pub.publish(img_feat);

    return;

}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "ImageFeatureExtractor")
    ros::NodeHandle n;

    std::string prototxt_path = "";
    std::string model_path = "./models/CNN.onnx";

    input_cvt = new imageConverter( 64, 64 );

    std::string cam_sub = "/camera/depth/image_rect_raw";
    std::string img_feat_pub = "/image_features";

    net = CNN::Create(prototxt_path, model_path);

    ros::Subscriber img_sub = n.subscribe(cam_sub, 4, depth_feat_extrac);
    ros::Publisher img_feat_pub = n.advertise<std_msgs::Float32MultiArray>(img_feat_pub, 1);

    ros::spin();

    delete net;
    delete input_cvt;
    delete output_features;
    delete temp;

    return 0;
}