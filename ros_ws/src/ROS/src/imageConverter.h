
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

        bool Resize ( uint8_t width, uint8_t height, const cv_bridge::CvImagePtr& input );

        bool assign( const sensor_msgs::ImageConstPtr& input );

        inline size_t GetInputSize() const { return mInputSize; }

        inline float* GetOutputCPU() const { return mOutputCPU; }

        inline uint* GetInputCPU() const { return mInputCPU; }
    
    private:

            size_t mInputSize;

            float* mOutputCPU;

            uint* mInputCPU;
            
            uint8_t req_height;
            uint8_t req_width;
};

#endif
