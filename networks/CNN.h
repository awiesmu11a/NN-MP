
#ifndef __CNN_H__
#define __CNN_H__

// do something about jetson-utils since tensorNet has dependencies on it - build it recurecisely along with other packages

#include "tensorNet.h"

#define IMGFEAT_NET_DEFAULT_INPUT   "images"

#define IMGFEAT_NET_DEFAULT_OUTPUT  "output"


class CNN : public tensorNet
{
public:

    static CNN* Create( const char* prototxt_path, const char* model_path, const char* input_blob = IMGFEAT_NET_DEFAULT_INPUT,
                        const Dims3& input_dim= Dims3(1, 64, 64), const char* output = IMGFEAT_NET_DEFAULT_OUTPUT, uint32_t maxBatchSize = 4,
                        precisionType precision = TYPE_FP32, deviceType device = DEVICE_GPU, bool allowGPUFallback = true );

	
	virtual ~CNN();

	bool Process();

protected:

	CNN();
};

#endif
