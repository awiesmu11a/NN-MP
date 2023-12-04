
// look into which headers and libraries to be used
#include "CNN.h"

CNN::CNN() : tensorNet()
{

}

CNN::~CNN()
{
    
}


CNN* CNN::Create( const char* prototxt, const char* model, const char* input_blob, 
                            const Dims3& input_dim, const char* output_blob,
                            uint32_t maxBatchSize, precisionType precision, deviceType device, bool allowGPUFallback )
{
    CNN* net = new CNN();

    // See if can be used
    // calibrator - nvInfer1::IInt8Calibrator* - No use unless we want th eprocessing to be done in int8
    // cudaStream_t stream - currently no use as all the stream manipulation is taken care
    // Can access the inputs from one of the variable in the constructor

    std::vector<std::string> outputs;
    outputs.push_back(output_blob);

    net->LoadNetwork( prototxt, model, NULL, input_blob, input_dim, outputs, maxBatchSize, 
                        precision, device, allowGPUFallback);
    
    return net;
}

// Image datatype - uint8*
// imageformat - IMAGE_GRAY8
// Also add an argument mentioning the history of images - Not sure where to include (in the batchsize not in the input size)
bool CNN::Process()
{
    // PROFILER_BEGIN(PROFILER_PREPROCESS)
    //  look if any form we need to remap the image from [0, 255] to [0, 1] - no need
    // Also verify if we need to keep the processor in sync - no need as it is by default
    PROFILER_BEGIN(PROFILER_NETWORK);
    if( !ProcessNetwork() ) return false;
    PROFILER_END(PROFILER_NETWORK);

    // update the pointer to output features
    // See according to tensorrt version how is context defined for an engine and checkout the execute or enqueue function
    return true;
}
