
#include <queue>
using std::queue;

#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/MultiArrayDimension.h>
#include <sensor_msgs/Image.h>

#include "tensorNet.h"
#include "imageConverter.h"
#include "CNN.h"

// add the library for cudaMappedMemory.h

CNN* net = NULL;
input_cvt = NULL;

queue<float32_t*> input_queue = new queue<float32_t*>;
float32_t* output_features = NULL;

// Look into freeing the memory appropriately and also temporary variables being defined

// Can change the batch of input images using one of the variables in the constructor
// use pointers and addresses to pass the data from sensors to the network
// Only thing to keep in mind is the datatype of the input and output of the network

void get_output( queue<float32_t*> &input )
{
    queue<float32_t*> temp_input = input;

    for (in i = 0; i < 4; i++)
    {
        float32_t* temp = temp_input.pop();
        size_t offset = i * 64 * 64 * sizeof(float32_t);
        cudaMemcpy(net.mInputs[0].CUDA + offset, temp, 64 * 64 * sizeof(float32_t), cudaMemcpyDeviceToDevice);
    }

    if ( !net->Process() )
    {
        ROS_ERROR("Image feature extraction failed");
        return;
    }

    return net.mOutputs[0].CPU;    
}

void depth_feat_extrac(const sensor_msgs::ImageConstPtr msg)
{
    if ( !input || !input_cvt->Convert(msg) )
    {
        ROS_ERROR("Image conversion failed");
        return;
    }

    // Add the input to the queue
    if ( input_queue.size() < 4 )
    {
        input_queue.push(input_cvt.mOutputGPU);
        return;
    }
    else
    {
        input_queue.pop();
        input_queue.push(input_cvt.mOutputGPU);
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

    std::string network = "CNN";
    std::string prototxt_path = "";
    std::string model_path = "./models/CNN.onnx";

    std::string input_blob = IMGFEAT_NET_DEFAULT_INPUT;
    std::string output_blob = IMGFEAT_NET_DEFAULT_OUTPUT;
    const Dims3& input_dim = (1, 64, 64);

    input_cvt = new imageConverter( int width = 64, int height = 64 );

    std::string cam_sub = "/camera/depth/image_rect_raw";
    std::string img_feat_pub = "/image_features";

    net = CNN::Create(prototxt_path, model_path, input_blob, input_dim, output_blob, maxBatchSize = 4);

    ros::Subscriber img_sub = n.subscribe(cam_sub, 4, depth_feat_extrac);
    ros::Publisher img_feat_pub = n.advertise<std_msgs::Float32MultiArray>(img_feat_pub, 1);

    ros::spin();
    // free resources from CUDA
    delete net;
    delete input_cvt;
    delete output_features;
    delete input_queue;

    return 0;
}