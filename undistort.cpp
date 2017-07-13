#include "undistort.h"

bool CUndistort::readParams()
{
    ifstream in;
    in.open(calibResultPath+"calibResult.txt", ios::in);
    in>>K.at<float>(0, 0);
    in>>K.at<float>(1, 1);
    in>>K.at<float>(0, 2);
    in>>K.at<float>(1, 2);

    in>>discoeff.at<float>(0,0);
    in>>discoeff.at<float>(1,0);
    in>>discoeff.at<float>(2,0);
    in>>discoeff.at<float>(3,0);
    in.close();
    return true;
}

bool CUndistort::undistProcess()
{
    //***************»û±äÐ£Õý****************//
    R=Mat::eye(Size(3, 3),CV_32FC1);
    Mat mapx, mapy;
    Mat srcImg=imread(srcImgPath+"test.jpg");
    pyrDown(srcImg, srcImg);
    pyrDown(srcImg, srcImg);
    Mat dstImg;
    fisheye::initUndistortRectifyMap(K, discoeff, R, K, Size(srcImg.cols, srcImg.rows),CV_32FC1, mapx, mapy);
    remap(srcImg, dstImg, mapx, mapy, CV_INTER_LINEAR);
    imshow("win", dstImg);
    waitKey(0);

    return true;
}

void CUndistort::run()
{
    bool readSuccess=readParams();
    undistProcess();
}

