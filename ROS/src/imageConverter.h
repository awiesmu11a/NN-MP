
#ifndef __ROS_IMAGE_CONVERTER_H__
#define __ROS_IMAGE_CONVERTER_H__

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

// Here we only need to convert ROS messages to matrix format

class imageConverter
{
    public:

        imageConverter(uint8_t width, uint8_t height);

        ~imageConverter();

        void Free();

        bool Convert( const sensor_msgs::ImageConstPtr& input );

        bool Resize ( uint8_t width, uint8_t height, const sensor_msgs::ImageConstPtr& input );

        bool cudaAssign( uint8_t width, uint8_t height, const sensor_msgs::ImageConstPtr& input );
    
    private:

            size_t mInputSize;
    
            float32_t* mOutputGPU; 
            float32_t* mOutputCPU;

            void* mInputCPU;
            void* mInputGPU;
            
            uint8_t req_height;
            uint8_t req_width;
}

#endif