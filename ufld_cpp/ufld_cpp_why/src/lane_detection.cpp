#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <fstream>
#include <sys/time.h>

#include "laneDetectDll.h"

using namespace std;
using namespace cv;

#define GPU


int main(void)
{
	struct timeval tv1,tv2;
	long long T;
	char *modelpath = "laneDetection.pt";
	Mat frame = imread("data/lane.jpg");
	TvPointList lanePoints_;
    laneDetectionInit(modelpath,&lanePoints_);


	//开始计时
	gettimeofday(&tv1, NULL);
	lanePoints_.num = 0;
	laneDetectionApi(frame.data, &lanePoints_);
	for (int i = 0; i < lanePoints_.num; i++) {
		cv::Point2f ImgPt(lanePoints_.point[i].x, lanePoints_.point[i].y);
		cv::circle(frame, ImgPt, 5, cv::Scalar(0, 0, 255), 2);
	}
	//结束计时
	gettimeofday(&tv2, NULL);
	//计算用时
	T = (tv2.tv_sec - tv1.tv_sec) * 1000 + (tv2.tv_usec - tv1.tv_usec) / 1000;
	cout << T << "ms" <<endl;

    imwrite("data/result_lane_why.jpg", frame);

	laneDetectionRelease(&lanePoints_);
    return 0;
}
