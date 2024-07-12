
#include "CNN.h"

CNN::CNN() : tensorNet()
{

}

CNN::~CNN()
{
    
}


CNN* CNN::Create( const char* prototxt, const char* model, uint32_t maxBatchSize, const Dims3& input_dim, 
                const char* input_blob, const char* output_blob, precisionType precision, deviceType device, bool allowGPUFallback )
{
    CNN* net = new CNN();

    std::vector<std::string> outputs;
    outputs.push_back(output_blob);

    net->LoadNetwork( prototxt, model, NULL, input_blob, input_dim, outputs, maxBatchSize, 
                        precision, device, allowGPUFallback);

    return net;
}

// Image datatype - uint8*
// imageformat - IMAGE_GRAY8

bool CNN::Process()
{
    PROFILER_BEGIN(PROFILER_NETWORK);
    if( !ProcessNetwork() ) return false;
    PROFILER_END(PROFILER_NETWORK);

    return true;
}
