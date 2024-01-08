
#ifndef __ROS_IMAGE_CONVERTER_H__
#define __ROS_IMAGE_CONVERTER_H__

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <sensor_msgs/Image.h>

class imageConverter
{
    public:

        imageConverter(uint8_t width = 64, uint8_t height = 64);

        ~imageConverter();

        void Free();

        bool Convert( const sensor_msgs::ImageConstPtr& input );

        bool Resize ( uint8_t width, uint8_t height, const sensor_msgs::ImageConstPtr& input );

        bool cudaAssign( uint8_t width, uint8_t height, const sensor_msgs::ImageConstPtr& input );

        inline size_t GetInputSize() const { return mInputSize; }

        inline float32_t* GetOutputGPU() const { return mOutputGPU; }

        inline uint8_t* GetInputGPU() const { return mInputGPU; }

        inline float32_t* GetOutputCPU() const { return mOutputCPU; }

        inline uint8_t* GetInputCPU() const { return mInputCPU; }
    
    private:

            size_t mInputSize;
    
            float32_t* mOutputCPU;
            float32_t* mOutputGPU;

            uint8_t* mInputCPU;
            uint8_t* mInputGPU;
            
            uint8_t req_height;
            uint8_t req_width;
}

#endif