
#ifndef __FCN_H__
#define __FCN_H__

#include "tensorNet.h"

#define FCN_NET_DEFAULT_INPUT "input"

#define FCN_NET_DEFAULT_OUTPUT "output"

class FCN : public tensorNet
{
public:

    static FCN* Create(const char* prototxt_path, const char* model_path, 
                        const Dims3& input_dim = Dims3(1, 12, 1), uint32_t maxBatchSize = 1,
                        const char* input_blob=FCN_NET_DEFAULT_INPUT, const char* output_blob = FCN_NET_DEFAULT_OUTPUT,
                        precisionType precision=TYPE_FP32, deviceType device = DEVICE_GPU, bool allowGPUFallback = true);
    
    virtual ~FCN();

    bool Process();

protected:

    FCN();
};


#endif
