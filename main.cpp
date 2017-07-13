#include <opencv.hpp>
#include <string>
#include <iostream>
#include "calibration.h"
#include "undistort.h"

using namespace std;
using namespace cv;

int main()
{
    string patternImgPath="data/pattern_images/";
    string calibResultPath="data/results/";
    string srcImgPath="data/source_images/";
    Size boardSize=Size(9, 6);
    CCalibration calibration(patternImgPath, calibResultPath, boardSize);
    calibration.run();
    CUndistort undistort(srcImgPath, calibResultPath);
    undistort.run();
}