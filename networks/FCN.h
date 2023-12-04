
#ifndef __FCN_H__
#define __FCN_H__

#include "tensorNet.h"

#define FCN_NET_DEFAULT_INPUT "vectors"

#define FCN_NET_DEFAULT_OUTPUT "outputs"

// Confirm the Dims2 before building the package


class FCN : public tensorNet
{
public:

    static FCN* Create(const char* prototxt_path, const char* model_path, const char* input_blob=FCN_NET_DEFAULT_INPUT,
                        const Dims2& input_dim = Dims2(12, 1), const char* output_blob = FCN_NET_DEFAULT_OUTPUT,
                        uint32_t maxBatchSize = 1, precisionType precision=TYPE_FP32, deviceType device = DEVICE_GPU, bool allowGPUFallback = true);
    
    virtual ~FCN();

    bool Process();

protected:

    FCN();
};


#endif