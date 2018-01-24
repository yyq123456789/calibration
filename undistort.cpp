#include "undistort.h"

bool CUndistort::readParams()
{
    ifstream in;
    in.open(calibResultPath+"calibResult.txt", ios::in);
    in>>K.at<float>(0, 0);
    in>>K.at<float>(1, 1);
    in>>K.at<float>(0, 2);
    in>>K.at<float>(1, 2);
	in >> discoeff.at<float>(0, 0);
	in >> discoeff.at<float>(1, 0);
	in >> discoeff.at<float>(2, 0);
	in >> discoeff.at<float>(3, 0);
    in.close();
    return true;
}

bool CUndistort::undistProcess()
{
    //***************»û±äÐ£Õý****************//
    R=Mat::eye(Size(3, 3),CV_32FC1);
    Mat mapx, mapy;
    Mat srcImg=imread(srcImgPath);
    Mat dstImg;
    cv::initUndistortRectifyMap(K, discoeff, R, K, srcImg.size(),CV_32FC1, mapx, mapy);
    remap(srcImg, dstImg, mapx, mapy, CV_INTER_LINEAR);
	cv::resize(dstImg, dstImg, cv::Size(), 0.25, 0.25, CV_INTER_LINEAR);
	cv::namedWindow("show", 1);
    imshow("show", dstImg);
    waitKey(0);

    return true;
}

void CUndistort::run()
{
    bool readSuccess=readParams();
    undistProcess();
}

