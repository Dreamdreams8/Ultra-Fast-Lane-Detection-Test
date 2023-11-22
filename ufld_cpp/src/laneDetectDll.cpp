#include <torch/script.h>
#include "laneDetectDll.h"
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <chrono>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
using namespace std;
using namespace cv;
using namespace torch::indexing;
torch::jit::script::Module detectModule;


int laneDetectionInit(char *modelpath)
{   
    detectModule = torch::jit::load(modelpath);
    detectModule.to(torch::kCUDA);
    detectModule.to(torch::kHalf);
    detectModule.eval();
    //printf("load the model success/n");

    return 1;
}

int laneDetectionMallocInit( TvPointList *lanePoints)
{   
    lanePoints->point = (TvPoint *)malloc(sizeof(TvPoint) * NUMPOINT);
    lanePoints->num = 0;
    //printf("set malloc success/n");
    return 1;
}

int laneDetectionRelease(TvPointList *lanePoints)
{
    if(lanePoints->point)
    {
        free(lanePoints->point);
        lanePoints->point = NULL;
    }

    return 1;
}

std::vector<double> linspace(double start_in, double end_in, int num_in)
{
    std::vector<double> linspaced;

    double start = static_cast<double>(start_in);
    double end = static_cast<double>(end_in);
    double num = static_cast<double>(num_in);

    if (num == 0) { return linspaced; }
    if (num == 1) 
    {
        linspaced.push_back(start);
        return linspaced;
    }

    double delta = (end - start) / (num - 1);

    for(int i=0; i < num-1; ++i)
    {
      linspaced.push_back(start + delta * i);
    }
    linspaced.push_back(end); // I want to ensure that start and end
                            // are exactly the same as the input

    return linspaced;
}

std::vector<int> arrange(int num)
{
    std::vector<int> result;
    for (int i = 0; i < num; i++)
    {
        result.push_back(i);
    }
    return result;
}

int laneDetectionApi(unsigned char *srcimg, TvPointList *lanePoints)
{
    // detectModule = torch::jit::load("/apollo/laneDetect/test/laneDetection.pt");
    // detectModule.to(torch::kCUDA);
    // detectModule.to(torch::kHalf);
    // detectModule.eval();


    if(!srcimg)
    {
        return -1;
    }
    int img_w = 1920;
    int img_h = 1080; 
    cv::Mat src(img_h, img_w, CV_8UC3);
    cv::Mat dest;
    src.data = srcimg;

    // CV Resize
    cv::resize(src, dest, cv::Size(800, 288));
    cv::cvtColor(dest, dest, cv::COLOR_BGR2RGB);  // BGR -> RGB
    dest.convertTo(dest, CV_32FC3, 1.0f / 255.0f);  // normalization 1/255
    int culane_row_anchor[] = {121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287};
    //printf("-----------------run here 0\n");

    auto tensor_img = torch::from_blob(dest.data, {1, dest.rows, dest.cols, dest.channels()}).to(torch::kCUDA);

    printf("-----------------run here 1\n");
    tensor_img = tensor_img.permute({0, 3, 1, 2}).contiguous();  // BHWC -> BCHW (Batch, Channel, Height, Width)
    printf("-----------------run here 11\n");
    tensor_img = tensor_img.to(torch::kHalf);
    printf("-----------------run here 12\n");
    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(tensor_img);
    printf("-----------------run here 13\n");
    //std::cout<< "the detectModule" << detectModule<< std::endl;
    torch::jit::IValue output = detectModule.forward(inputs);

    printf("-----------------run here 14\n");
    torch::Tensor outputTensor = output.toTensor();
    //printf("-----------------run here 2\n");
    // Logic
    int cuLaneGriding_num = 200;
    std::vector<double> linSpaceVector = linspace(0, 800 - 1, cuLaneGriding_num);
    double linSpace = linSpaceVector[1] - linSpaceVector[0];
    // Remove 1
    outputTensor = outputTensor.squeeze(0);
    // Flip
    outputTensor = outputTensor.flip(1);

    //printf("-----------------run here 3\n");
    // Calculate SoftMax
    torch::Tensor prob = outputTensor.index({Slice(None, -1)}).softmax(0);
    //printf("-----------------run here 4\n");
    // Calculate idx
    std::vector<int> idx = arrange(cuLaneGriding_num + 1);  
    auto arrange_idx = torch::from_blob(idx.data(), {cuLaneGriding_num, 1, 1}).to(torch::kCUDA);
    outputTensor = outputTensor.argmax(0);
		
    //printf("-----------------run here 5\n");

    auto mult = prob * arrange_idx;

    auto loc = mult.sum(1);
    for (int i = 0; i < outputTensor.size(0); i++)
    {  
   			if (outputTensor[i][0].item<long>() == cuLaneGriding_num)
				{
					outputTensor[i][0] = 0;
				}	
   			if (outputTensor[i][1].item<long>() == cuLaneGriding_num)
				{
					outputTensor[i][1] = 0;
				}	
   			if (outputTensor[i][2].item<long>() == cuLaneGriding_num)
				{
					outputTensor[i][2] = 0;
				}	     
   			if (outputTensor[i][3].item<long>() == cuLaneGriding_num)
				{
					outputTensor[i][3] = 0;
				}	
    }
	torch::Tensor res = outputTensor;
    //printf("-----------------run here 6\n");
    int nump = 0;
    lanePoints->num = 0;
    for (int i = 0; i < outputTensor.size(1); i++)
    {
        for (int k = 0; k < outputTensor.size(0); k++)
        {
            if (outputTensor[k][i].item<long>() > 0)
            {
                long temp = outputTensor[k][i].item<long>();
                lanePoints->point[nump].x = outputTensor[k][i].item<long>()  * linSpace * img_w /800;
                lanePoints->point[nump].y = img_h * (float(culane_row_anchor[18-1-k])/288);
                lanePoints->num ++;
                nump ++;
            }
        }
    }
    //printf("-----------------run here 7\n");
    return 1;
}
