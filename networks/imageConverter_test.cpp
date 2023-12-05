
#include "imageConverter_test.h"

#include <jetson-utils/cudaMappedMemory.h>
#include <jetson-utils/imageFormat.h>
#include <opencv2/opencv.hpp>

// File used to resize the input image to the required which is 64x64
// Change the format to call the dimensions of the image in the openCV format

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

bool imageConverter::Convert( const cv::Mat &input )
{
    if ( !input )
    {
        sprintf("Invalid image input");
        return false;
    }

    mInputCPU = input.data;

    if ( input->width == req_width && input->height == req_height )
    {
        sprintf("Image size is as required so no conversion needed");

        std::memcpy(mOutputCPU, mInputCPU, imageFormatSize(IMAGE_GRAY8, input->width, input->height));

        cudaAllocMapped( (void**)&mInputCPU, (void**)&mInputGPU,
                        imageFormatSize(IMAGE_GRAY8, input->width, input->height) );
        cudaAllocMapped( (void**)&mOutputCPU, (void**)&mOutputGPU,
                        imageFormatSize(IMAGE_GRAY8, input->width, input->height) );
        
        mInputSize = imageFormatSize(IMAGE_GRAY8, input->width, input->height);

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

bool imageConverter::Resize( uint8_t width, uint8_t height, const cv::Mat &input )
{
    cv::Mat cv_resized_img;
    cv::resize(input, cv_resized_img, cv::Size(width, height));

    std::memcpy(mOutputCPU, cv_resized_img.data, imageFormatSize(IMAGE_GRAY8, width, height));

    cv_resized_img.release();

    return true;
}

bool imageConverter::cudaAssign( uint8_t width, uint8_t height, const cv::Mat &input )
{
    const size_t input_size = imageFormatSize( IMAGE_GRAY8, input->width, input->height );
    const size_t output_size = imageFormatSize( IMAGE_GRAY8, width, height );

    if( input_size != mInputSize )
    {
        Free();
        mInputCPU = input.data;
        
        if( !cudaAllocMapped( (void**)&mInputCPU, (void**)&mInputGPU, input_size ) ||
            !cudaAllocMapped( (void**)&mOutputCPU, (void**)&mOutputGPU, output_size ) )
        {
            printf("failed to allocate %zu bytes for input image\n", input_size);
            return false;
        }

        mInputSize = input_size;
    }

    input_size.clear();
    output_size.clear();

    return true;
}