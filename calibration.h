#include "opencv.hpp"
#include <string>
#include <fstream>
#include <iostream>

using namespace std;
using namespace cv;


class CCalibration
{
public:
    CCalibration(string patternImgPath, string CalibResultPath, Size boardSize)
    {
        this->patternImgPath=patternImgPath;
        this->calibResultPath=CalibResultPath;
        this->boardSize=boardSize;
    }
    ~CCalibration(){}

private:
    vector<Mat> patternImgList; 
    int imgHeight;
    int imgWidth;
    int imgNum;
    string patternImgPath;
    string calibResultPath;
    Size boardSize;
    Mat camK;
    Mat camDiscoeff;

private:
    bool evaluateCalibrationResult( vector<vector<Point3f>> objectPoints, vector<vector<Point2f>> cornerSquare, vector<int> pointCnts, vector<Vec3d> _rvec,  vector<Vec3d> _tvec, Mat _K, Mat _D, int count);

public:
    bool writeParams();
    bool readPatternImg();
    void calibProcess();
    void run();
};