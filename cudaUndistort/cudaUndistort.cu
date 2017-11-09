#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

#define PI 3.1415926

__global__ void cudaBuildMap(float *pCamK, float *pDistort, float *pInvNewCamK, float *pMapx, float *pMapy, int outImgW, int outImgH)
{
	const int tidx = blockDim.x*blockIdx.x + threadIdx.x;
	const int tidy = blockDim.y*blockIdx.y + threadIdx.y;
	if (tidx < outImgW && tidy < outImgH)
	{
		float k1 = pDistort[0];
		float k2 = pDistort[1];
		float p1 = pDistort[2];
		float p2 = pDistort[3];
		float k3, k4, k5, k6, s1, s2, s3, s4;
		k3 = k4 = k5 = k6 = s1 = s2 = s3 = s4 = 0;
		float fx = pCamK[0];
		float fy = pCamK[4];
		float u0 = pCamK[2];
		float v0 = pCamK[5];

		//ͼ������ϵ->���������ϵ
		float _x = tidx*pInvNewCamK[0] + tidy*pInvNewCamK[1] + pInvNewCamK[2];
		float _y = tidx*pInvNewCamK[3] + tidy*pInvNewCamK[4] + pInvNewCamK[5];
		float _w = tidx*pInvNewCamK[6] + tidy*pInvNewCamK[7] + pInvNewCamK[8];
		//��һ��
		float w = 1. / _w;
		float x = _x * w;
		float y = _y * w;
		//
		float x2 = x*x;
		float y2 = y*y;
		float r2 = x2 + y2;
		float _2xy = 2 * x*y;
		float kr = (1 + ((k3*r2 + k2)*r2 + k1)*r2) / (1 + ((k6*r2 + k5)*r2 + k4)*r2);
		float xd = (x*kr + p1*_2xy + p2*(r2 + 2 * x2) + s1*r2 + s2*r2*r2);
		float yd = (y*kr + p1*(r2 + 2 * y2) + p2*_2xy + s3*r2 + s4*r2*r2);

		float invProj = 1.;
		float u = fx*invProj*xd + u0;
		float v = fy*invProj*yd + v0;

		int mapIdx = tidy*outImgW + tidx;
		pMapx[mapIdx] = (float)u;
		pMapy[mapIdx] = (float)v;
	}
}

__global__ void cudaRemap (uchar* pSrcImg, uchar* pDstImg, float* pMapx, float* pMapy, int inWidth, int inHeight, int outWidth, int outHeight, int channels)
{
	const int tidx = blockDim.x*blockIdx.x + threadIdx.x;
	const int tidy = blockDim.y*blockIdx.y + threadIdx.y;
	if (tidx < outWidth && tidy < outHeight)
	{
		int mapIdx = tidy*outWidth + tidx;
		float u = pMapx[mapIdx];
		float v = pMapy[mapIdx];

		int u1 = floor(u);
		int v1 = floor(v);
		int u2 = u1 + 1;
		int v2 = v1 + 1;
		if (u1 >= 0 && v1 >= 0 && u2 < inWidth && v2 < inHeight)
		{
			float dx = u - u1;
			float dy = v - v1;
			float weight1 = (1 - dx)*(1 - dy);
			float weight2 = dx*(1 - dy);
			float weight3 = (1 - dx)*dy;
			float weight4 = dx*dy;

			int resultIdx = mapIdx * 3;
			for (int chan = 0; chan < channels; chan++)
			{
				pDstImg[resultIdx + chan] = uchar(weight1*pSrcImg[(v1*inWidth + u1) * 3 + chan]
					+ weight2*pSrcImg[(v1*inWidth + u2) * 3 + chan]
					+ weight3*pSrcImg[(v2*inWidth + u1) * 3 + chan]
					+ weight4*pSrcImg[(v2*inWidth + u2) * 3 + chan]);
			}
		}
	}
}

void main()
{
	cv::Mat srcImg = cv::imread("data/test.jpg");
	cv::Mat camK = cv::Mat::eye(3, 3, CV_32F);
	cv::Mat newCamK = cv::Mat::eye(3, 3, CV_32F);
	cv::Mat invCamK = cv::Mat::eye(3, 3, CV_32F);
	cv::Mat D = cv::Mat::zeros(4, 1, CV_32F);
	cv::Mat R = cv::Mat::eye(3, 3, CV_32F);

	int imgHeight = srcImg.rows;
	int imgWidth = srcImg.cols;
	int channels = srcImg.channels();
	int outImgHeight = imgHeight;
	int outImgWidth = imgWidth;
	cv::Mat undistortImg=cv::Mat(outImgHeight, outImgWidth, CV_8UC3);
	cv::Mat mapx = cv::Mat(outImgHeight, outImgWidth, CV_32F);
	cv::Mat mapy = cv::Mat(outImgHeight, outImgWidth, CV_32F);

	//��ת����R
	float rotation[9] = { 0.999956, 0.00837928, 0.00423897,
		- 0.00839916, 0.999954, 0.00469485, 
		- 0.00419943, - 0.00473025, 0.99998 };
	R = cv::Mat(3, 3, CV_32F, rotation);

	//�ڲξ���
	camK.at<float>(0, 2) = imgWidth / 2;
	camK.at<float>(1, 2) = imgHeight / 2;
	camK.at<float>(0, 0) = camK.at<float>(0, 2) / tan((120*PI/180) / 2);
	camK.at<float>(1, 1) = camK.at<float>(0, 0);
	camK.at<float>(2, 2) = 1;
	newCamK = camK.clone();
	invCamK = (newCamK*R.t()).inv(cv::DECOMP_LU);

	cudaError err;
	float *pCamKData = NULL;
	float *pInvNewCamKData = NULL;
	float *pDistortData = NULL;
	uchar *pSrcImgData = NULL;
	uchar *pDstImgData = NULL;
	float *pMapxData = NULL;
	float *pMapyData = NULL;
	{
		err = cudaMalloc(&pCamKData, 9 * sizeof(float));
		err = cudaMalloc(&pInvNewCamKData, 9 * sizeof(float));
		err = cudaMalloc(&pDistortData, 4 * sizeof(float));
		err = cudaMalloc(&pSrcImgData, imgHeight*imgWidth * sizeof(uchar)*channels);
		err = cudaMalloc(&pMapxData, outImgHeight*outImgWidth * sizeof(float));
		err = cudaMalloc(&pMapyData, outImgHeight*outImgWidth * sizeof(float));
		err = cudaMalloc(&pDstImgData, outImgHeight*outImgWidth * sizeof(uchar)*channels);
	}
	{
		err = cudaMemcpy(pCamKData, camK.data, 9 * sizeof(float), cudaMemcpyHostToDevice);
		err = cudaMemcpy(pInvNewCamKData, invCamK.data, 9 * sizeof(float), cudaMemcpyHostToDevice);
		err = cudaMemcpy(pDistortData, D.data, 4 * sizeof(float), cudaMemcpyHostToDevice);
		err = cudaMemcpy(pMapxData, mapx.data, outImgHeight*outImgWidth * sizeof(float), cudaMemcpyHostToDevice);
		err = cudaMemcpy(pMapyData, mapy.data, outImgHeight*outImgWidth * sizeof(float), cudaMemcpyHostToDevice);
		err = cudaMemcpy(pSrcImgData, srcImg.data, imgHeight*imgWidth * sizeof(uchar)*channels, cudaMemcpyHostToDevice);
	}

	dim3 block(16, 16);
	dim3 grid((imgWidth + block.x - 1) / block.x, (imgHeight + block.y - 1) / block.y);
	cudaBuildMap << <grid, block >> > (pCamKData, pDistortData, pInvNewCamKData, pMapxData, pMapyData, outImgWidth, outImgHeight);
	cudaThreadSynchronize();
	cudaRemap << <grid, block >> > (pSrcImgData, pDstImgData, pMapxData, pMapyData, imgWidth, imgHeight, outImgWidth, outImgHeight, channels);
	cudaThreadSynchronize();

	err = cudaGetLastError();
	err = cudaMemcpy(undistortImg.data, pDstImgData, outImgWidth*outImgHeight * sizeof(uchar)*channels, cudaMemcpyDeviceToHost);

	/*cv::initUndistortRectifyMap(camK, D, R, camK, srcImg.size(), CV_32F, mapx, mapy);
	cv::remap(srcImg, undistortImg, mapx, mapy, CV_INTER_LINEAR);*/
	
	cv::imwrite("data/undistortImg.jpg", undistortImg);
}