#include "opencv.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace cv;

int evaluateCalibrationResult( vector<vector<Point3f>> objectPoints, vector<vector<Point2f>> cornerSquare, vector<int> pointCnts,
    vector<Vec3d> _rvec,  vector<Vec3d> _tvec, Mat _K, Mat _D, int count);

int writeParams(string fileName,  Mat& matData);