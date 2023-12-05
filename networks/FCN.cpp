
#include "FCN.h"

// One problem could be due to defining nature of Dims2
// See if we need to #define the dimensions for Dims2 like in Dims3

FCN::FCN() : tensorNet()
{

}

FCN::~FCN()
{

}

FCN* FCN::Create( const char* prototxt, const char* model, const Dims2& input_dim, uint32_t maxBatchSize,
                const char* input_blob, const char* output_blob, 
                precisionType precision, deviceType device, bool allowGPUFallback)
{
    FCN* net = new FCN();

    std::vector<std::string> outputs;
    outputs.push_back(output_blob);

    net->LoadNetwork( prototxt, model, NULL, input_blob, input_dim, outputs,
                    maxBatchSize, precision, device, allowGPUFallback);
    
    outputs.clear();
    return net;
}

FCN* FCN::Process()
{
    PROFILER_BEGIN(PROFILER_NETWORK);
    if( !ProcessNetwork() ) return false;
    PROFILER_END(PROFILER_NETWORK);

    return true;
}

