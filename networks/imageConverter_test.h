

#ifndef __TEST_IMAGE_CONVERTER_H__
#define __TEST_IMAGE_CONVERTER_H__

#include <opencv2/opencv.hpp>

class imageConverter
{
    public:

        imageConverter(uint8_t width = 64, uint8_t height = 64);

        ~imageConverter();

        void Free();

        bool Convert( const cv::Mat &input );

        bool Resize ( uint8_t width, uint8_t height, const cv::Mat &input );

        bool cudaAssign( uint8_t width, uint8_t height, const cv::Mat &input );
    
    private:
    
            size_t mInputSize;
    
            float32_t* mOutputCPU;
            float32_t* mOutputGPU;

            uint8_t* mInputCPU;
            uint8_t* mInputGPU;
            
            uint8_t req_height;
            uint8_t req_width;
};

#endif
