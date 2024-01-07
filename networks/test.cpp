
#include <iostream>
#include <fstream>
#include <sstream>
#include <queue>

#include <opencv2/opencv.hpp>

#include "imageConverter_test.h"
#include "CNN.h"
#include "FCN.h"


bool CNN_test(const char* input) 
{
    imageConverter* input_cvt = new imageConverter( 64, 64 );

    cv::Mat image = cv::imread(input, cv::IMREAD_GRAYSCALE);
    
    if ( !input_cvt->Convert( image ) )
    {
        printf("Failed to convert input image");
        return false;
    }

    const char* prototxt_path = "";
    const char* model_path = "./models/CNN.onnx";

    CNN* net = CNN::Create( prototxt_path, model_path );

    size_t offset = input_cvt->GetInputSize();

    for( int i = 0; i < 4; i++ )
    {
        cudaMemcpy( net->GetInputPtr(0) + offset * i, input_cvt->GetOutputGPU(), input_cvt->GetInputSize(), cudaMemcpyDeviceToDevice );
    }

    if( !net->Process() )
    {
        printf("Failed to process CNN");
        return false;
    }

    printf("Got a feature vector of size %u", net->GetOutputSize(0));

    delete net;
    delete input_cvt;
    delete[] prototxt_path;
    delete[] model_path;

    return true;
}

std::vector<std::vector<float>> extract_csv(const char* input)
{
    std::ifstream file(input);

    std::vector<std::vector<float>> data;

    std::string line = "";

    while (getline(file, line))
    {
        std::vector<float> vec;
        std::stringstream ss(line);

        float i;

        while (ss >> i)
        {
            vec.push_back(i);

            if (ss.peek() == ',')
            {
                ss.ignore();
            }
        }

        data.push_back(vec);
    }

    file.close();

    return data;
}

bool FCN_test(const char* input) 
{
    std::vector<std::vector<float>> data = extract_csv(input);
    const Dims2& input_dim = Dims2(data.size(), data[0].size());
    uint8_t input_size = data.size() * data[0].size() * sizeof(float);

    const char* prototxt_path = "";
    const char* model_path = "./models/FCN.onnx";

    FCN* net = FCN::Create( prototxt_path, model_path, input_dim );

    cudaMalloc( &net->GetInputPtr(0), input_size );

    cudaMemcpy( net->GetInputPtr(0), data, input_size, cudaMemcpyHostToDevice );

    if( !net->Process() )
    {
        printf("Failed to process FCN");
        return false;
    }

    printf("Got a feature vector of size %u", net->GetOutputSize(0));

    delete net;
    delete[] prototxt_path;
    delete[] model_path;
    delete output;
    data.clear();

    return true;
}

int main (int argc, char** argv)
{
    const char* input;
    if ( argc < 1) {
        printf("Please provide an input to process");
        return 0;
    }

    if (argc < 2 ) {
        printf("Please provide the type of input");
        printf("-i for image \n");
        printf("-v for vector \n");
        return 0;
    }

    if (argv[2] == "-i")
    {
        input = argv[1];
        if ( !CNN_test(input) )
        {
            printf("Failed to process CNN");
            return 0;
        }

    }

    if (argv[2] == "-v")
    {
        input = argv[1];
        if ( !FCN_test(input) )
        {
            printf("Failed to process FCN");
            return 0;
        }
    }

    delete[] input;

    return 0;

}