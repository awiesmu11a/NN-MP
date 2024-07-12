
#ifndef __CNN_H__
#define __CNN_H__

#include "tensorNet.h"

#define IMGFEAT_NET_DEFAULT_INPUT   "input"

#define IMGFEAT_NET_DEFAULT_OUTPUT  "output"


class CNN : public tensorNet
{
public:

    static CNN* Create( const char* prototxt_path, const char* model_path, uint32_t maxBatchSize = 1,
                        const Dims3& input_dim= Dims3(4, 64, 64), const char* input_blob = IMGFEAT_NET_DEFAULT_INPUT,
                        const char* output = IMGFEAT_NET_DEFAULT_OUTPUT, precisionType precision = TYPE_FP32, 
                        deviceType device = DEVICE_GPU, bool allowGPUFallback = true );

	
	virtual ~CNN();

	bool Process();

protected:

	CNN();
};

#endif
