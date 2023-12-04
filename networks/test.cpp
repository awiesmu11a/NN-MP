
#include "CNN.h"

int main (int argc, char** argv)
{
    const char* prototxt_path = "";
    const char* model_path = "./models/temp.onnx";
    
    CNN* net = CNN::Create( prototxt_path, model_path );

    printf("---------------Created the CNN object successfully-----------------------");

    delete net;
}
