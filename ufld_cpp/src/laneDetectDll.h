#ifndef _LANEDETECT_
#define _LANEDETECT_


#ifdef __cplusplus
extern "C"
{
#endif

#define NUMPOINT 100
// 点数据结构
typedef struct TvPoint
{
	short x;
	short y;
}TvPoint;

// 点列表结构
typedef struct TvPointList
{//10B
	short num;
	TvPoint *point;
}TvPointList;


/*************************************
 * 函数名称：laneDetectionInit
 * 函数功能：模型初始化
 * 输入    ：modelpath 模型路径
 * 输入    : lanePoints 检测点
 * ***********************************/
int laneDetectionInit(char *modelpath);

/*************************************
 * 函数名称：laneDetectionMallocInit
 * 函数功能：内存分配
 * 输入    : lanePoints 检测点
 * ***********************************/
int laneDetectionMallocInit( TvPointList *lanePoints);

/*************************************
 * 函数名称：laneDetectionApi
 * 函数功能：车道线检测
 * 输入    ：src 采集的图像
 * 输出    ：lanePoints 检测点
 * ***********************************/
int laneDetectionApi(unsigned char *src, TvPointList *lanePoints);

/*************************************
 * 函数名称：laneDetectionRelease
 * 函数功能：释放内存
 * 输入    lanePoints
 * ***********************************/
int laneDetectionRelease(TvPointList *lanePoints);

#ifdef __cplusplus
}
#endif

#endif