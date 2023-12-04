
#include "imageConverter.h"

#include <jetson-utils/cudaMappedMemory.h>
#include <jetson-utils/imageFormat.h>
#include <opencv2/opencv.hpp>

imageConverter::imageConverter( uint8_t width, uint8_t height )
{
    mInputSize = 0;

    mOutputCPU = NULL;
    mOutputGPU = NULL;

    mInputCPU = NULL;
    mInputGPU = NULL;    

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
        mInputGPU = NULL;
    }

    if( mOutputCPU != NULL )
    {
        CUDA(cudaFreeHost(mOutputCPU));
        mOutputCPU = NULL;
        mOutputGPU = NULL;
    }

    mInputSize = 0;

}

bool imageConverter::Convert( const sensor_msgs::ImageConstPtr& input )
{
    if ( !input )
    {
        sprintf("Invalid image input");
        return false;
    }

    if ( input->width == req_width && input->height == req_height )
    {
        sprintf("Image size is as required so no conversion needed");
        // Add more here assigning to CUDA and CPU memory
        return true;

    }

    if ( !cudaAssign( req_width, req_height, input ) ) { return false; }

    if ( !Resize( req_width, req_height, input ) )
    {
        sprintf("Image size is not as required and resizing failed");
        return false;
    }

    return true;

}

bool imageConverter::Resize( uint8_t width, uint8_t height, const sensor_msgs::ImageConstPtr& input )
{
    cv::Mat cv_img = cv::imdecode(cv::Mat(input->height, input->width, CV_8UC1, const_cast<uint8_t*>(input->data.data())),
                                    cv::IMREAD_GRAYSCALE);
    cv::Mat cv_resized_img;
    cv::resize(cv_img, cv_resized_img, cv::Size(width, height));

    std::memcpy(mOutputCPU, cv_resized_img.data, imageFormatSize(IMAGE_GRAY8, width, height));

    return true;
}

bool imageConverter::cudaAssign( uint8_t width, uint8_t height, const sensor_msgs::ImageConstPtr& input )
{
    const size_t input_size = imageFormatSize( IMAGE_GRAY8, input->width, input->height );
    const size_t output_size = imageFormatSize( IMAGE_GRAY8, width, height );

    if( input_size != mInputSize )
    {
        Free();
        if( !cudaAllocMapped( (void**)&mInputCPU, (void**)&mInputGPU, input_size ) ||
            !cudaAllocMapped( (void**)&mOutputCPU, (void**)&mOutputGPU, output_size ) )
        {
            printf("failed to allocate %zu bytes for input image\n", input_size);
            return false;
        }

        mInputSize = input_size;
    }

    return true;
}