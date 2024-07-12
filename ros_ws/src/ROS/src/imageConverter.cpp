
#include "imageConverter.h"

#include <jetson-utils/cudaMappedMemory.h>
#include <jetson-utils/imageFormat.h>
#include <opencv2/opencv.hpp>

#include <sensor_msgs/Image.h>

imageConverter::imageConverter( uint8_t width, uint8_t height )
{
    mInputSize = 0;

    mOutputCPU = NULL;   

    mInputCPU = NULL;

    req_width = width;
    req_height = height;
}

imageConverter::~imageConverter()
{
    Free();
}

void imageConverter::Free()
{
    if( mInputCPU != NULL )
    {
        CUDA(cudaFreeHost(mInputCPU));
        mInputCPU = NULL;
    }

    if( mOutputCPU != NULL )
    {
        CUDA(cudaFreeHost(mOutputCPU));
        mOutputCPU = NULL;
    }

    mInputSize = 0;

}

bool imageConverter::Convert( const sensor_msgs::ImageConstPtr& input )
{
    if ( !input )
    {
        printf("Invalid image input");
        return false;
    }

    cv_bridge::CvImagePtr cv_ptr;

    cv_ptr = cv_bridge::toCvCopy(input, sensor_msgs::image_encodings::TYPE_8UC1);

    std::memcpy(mInputCPU, cv_ptr->image.data, imageFormatSize(IMAGE_GRAY8, cv_ptr->image.rows, cv_ptr->image.cols));

    if ( !Resize( req_width, req_height, cv_ptr ) )
    {
        printf("Image size is not as required and resizing failed");
        return false;
    }

    return true;

}

bool imageConverter::Resize( uint8_t width, uint8_t height, const cv_bridge::CvImagePtr& input )
{
    cv::Mat cv_resized_img;
    cv::Mat cv_img = cv::Mat(input->image.rows, input->image.cols, CV_8UC1, input->image.data);

    cv::resize(cv_img, cv_resized_img, cv::Size(64, 64));
    cv_resized_img.convertTo(cv_resized_img, CV_32FC1);

    std::memcpy(mOutputCPU, cv_resized_img.data, imageFormatSize(IMAGE_GRAY32F, 64, 64));

    return true;
}

bool imageConverter::assign( const sensor_msgs::ImageConstPtr& input )
{
    cv_bridge::CvImagePtr cv_ptr;

    cv_ptr = cv_bridge::toCvCopy(input, sensor_msgs::image_encodings::TYPE_8UC1);

    CUDA(cudaMallocHost((void**)&mInputCPU, imageFormatSize(IMAGE_GRAY8, cv_ptr->image.rows, cv_ptr->image.cols)));

    CUDA(cudaMallocHost((void**)&mOutputCPU, imageFormatSize(IMAGE_GRAY32F, req_height, req_width)));

    mInputSize = imageFormatSize(IMAGE_GRAY8, cv_ptr->image.rows, cv_ptr->image.cols);

    return true;
}