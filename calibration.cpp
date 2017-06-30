#include "opencv.hpp"
#include "calibration.h"
#include <fstream>
#include <iostream>

using namespace std;
using namespace cv;

#define ImgNum  11

 //摄像机标定

int main()
{
//***************摄像机标定****************//
    double time0=getTickCount();
    string calibFile="Results/calibResult.txt";
    ofstream fileout(calibFile, ios::trunc);//清空原有数据
    fileout.close();
    int imgCount = ImgNum;//棋盘格图片个数
    Size board_size = Size(9,6);//角点个数
    vector<Point2f> corners;//存储一幅棋盘图中的所有角点二维坐标
    vector<vector<Point2f>> corners_Seq;//存储所有棋盘图角点的二维坐标
    vector<Mat> image_Seq;//存储所有棋盘图
    int successImgNum=0;
    int count=0;
    cout<<"********开始提取角点！********"<<endl;
    for (int i=0; i<imgCount; i++)
    {
	    cout<<"Image#"<<i+1<<"......."<<endl;
	    string imgFileName;
	    stringstream StrStm;
	    string filenum;
	    StrStm<<i+1;
	    StrStm>>filenum;
	    imgFileName ="./Images/Pattern/img" +filenum+ ".jpg";
	    Mat image = imread(imgFileName);   //根据文件名依次读取棋盘图
        pyrDown(image, image);
        pyrDown(image, image);
	    /**********************提取角点*************************/
	    Mat greyImg;
	    cvtColor(image, greyImg, CV_RGB2GRAY);
	    bool patternfound= findChessboardCorners(image, board_size,
            corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE+CALIB_CB_FAST_CHECK);
	    if (!patternfound)
	    {
		    cout<<"Can not find chess board corners!\n"<<endl;
		    continue;
		    exit(1);
	    }
	    else
	    {
		    /************************亚像素精确化******************************/
		    cornerSubPix(greyImg, corners, Size(11, 11), Size(-1,-1) , TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
		    /************************绘制检测到的角点并保存******************************/
		    Mat tempImg = image.clone();
		    for (int j=0; j< corners.size(); j++)
		    {
			    circle(tempImg, corners[j], 10, Scalar(0,0,255), 2, 8, 0);
		    }
		    string tempFileName;
		    stringstream tempStrStm;
		    string tempfilenum;
		    tempStrStm<<i+1;
		    tempStrStm>>filenum;
		    tempFileName ="./Images/Pattern/CornerImage/img" + filenum +"_corner.jpg";
		    imwrite(tempFileName, tempImg);//将角点绘制出后保存
		    cout<<"Corner_Image#"<<i+1<<"......"<<endl;

		    count +=corners.size();//所欲棋盘图中的角点个数
		    successImgNum++;//成功提取角点的棋盘图个数
		    corners_Seq.push_back(corners);
	    }
	    image_Seq.push_back(image);
    }
    cout<<"*******角点提取完成！******"<<endl;
	
    /**************************摄像机标定******************************/
    Size squre_size=Size(20,20);//棋盘格尺寸
    vector<vector<Point3f>> object_points;//所有棋盘图像的角点三维坐标
    vector<int> point_counts;
    /*初始化标定板上的三维坐标*/
    for(int n=0; n<successImgNum; n++)
    {
	    vector<Point3f> tempPointSet;//单幅棋盘图像的所有角点
	    for (int i=0; i<board_size.height; i++)
	    {
		    for (int j=0; j<board_size.width; j++)
		    {
			    Point3f tempPoint;//单个角点的三维坐标
			    tempPoint.x = i * squre_size.width;
			    tempPoint.y = j* squre_size.height;
			    tempPoint.z = 0;
			    tempPointSet.push_back(tempPoint);
		    }
	    }
	    object_points.push_back(tempPointSet);
    }
    for (int i=0; i<successImgNum; i++)
    {
	    point_counts.push_back(board_size.width * board_size.height);//保存每一幅棋盘图的角点个数
    }
    /***开始标定***/
    cout<<"*****开始标定!******"<<endl;
    Size image_size=image_Seq[0].size();
    Mat intrinsic_matrix; //内参矩阵
    Mat distortion_coeffs;//畸变系数
    vector<Vec3d> rotation_vectors;//旋转向量
    vector<Vec3d> translation_vectors;//平移向量
    int flags=0;
    flags |= fisheye::CALIB_RECOMPUTE_EXTRINSIC;
    flags |= fisheye::CALIB_CHECK_COND;
    flags |= fisheye::CALIB_FIX_SKEW;
    fisheye::calibrate(object_points, corners_Seq, image_size, intrinsic_matrix, distortion_coeffs, rotation_vectors, translation_vectors, flags, cv::TermCriteria(3, 20, 1e-6));
    cout<<"*****标定完成！*****"<<endl;
    double time1=getTickCount();
    cout<<"Calibration time :"<<(time1-time0)/getTickFrequency()<<"s"<<endl;

    /****************************对标定结果进行评价*****************************/
    evaluateCalibrationResult( object_points, corners_Seq, point_counts,  rotation_vectors,  translation_vectors, intrinsic_matrix, distortion_coeffs, imgCount);

    /*********************保存标定结果******************/
	    cout<<"*****开始保存标定结果*****"<<endl; 
    //writeParams(intrinsicFile, intrinsic_matrix );//保存内参矩阵
    //writeParams(disCoeffsFile, distortion_coeffs );//保存畸变系数
    writeParams( calibFile, intrinsic_matrix );
    writeParams( calibFile, distortion_coeffs );
    cout<<"******完成保存******"<<endl;
      
}

/******************************************************************************
 * 函数名称:    writeParams()
 * 函数描述:    将标定获取的参数（内参和畸变系数）输出到指定txt文件
 * 输    入:        matData               -参数矩阵
 * 输    出:        fileName              -文件名
 * 返 回 值:       0                         -成功
                        其他                    -失败
 ******************************************************************************/
int writeParams(string fileName,  cv::Mat& matData)
{
	int retVal=0;

	//打开文件
	ofstream foutFile( fileName, ios::app);
	if (0==foutFile.is_open())
	{
		cout<<"Open File Failed!"<<endl;
		return -1;
	}

	for (int i =0; i< matData.rows; i++)
	{
		for (int j=0; j< matData.cols; j++)
		{
			double data=matData.at<double>(i , j);
			foutFile<<data<<"    ";
		}
	}
    foutFile<<endl;    //每一行为一组数据
    foutFile.close();
	return retVal;
}
/******************************************************************************
 * 函数名称:    evaluateCalibrationResult()
 * 函数描述:    评价标定结果并保存评价结果
 * 输    入:        objectPoints               -三维角点坐标
                        cornerSquare             -二维角点坐标
                        pointCnts                   -角点个数
                        _rvec                        -旋转向量
                        _tvec                        -平移向量
                        _K                            -内参矩阵
                        _D                            -畸变系数
                        count                         -棋盘格图片个数
 * 输    出:        无
 * 返 回 值:       0                              -成功
                        其他                         -失败
 ******************************************************************************/
int evaluateCalibrationResult( vector<vector<Point3f>> objectPoints, vector<vector<Point2f>> cornerSquare, vector<int> pointCnts,  vector<Vec3d> _rvec,  vector<Vec3d> _tvec, Mat _K, Mat _D, int count)
{
	ofstream fout("./Results/evaluateCalibrationResult.txt");
	cout<<"******开始评价标定结果******"<<endl;
	double total_err = 0.0;//所有图像的平均误差和
	double err=0.0;//单幅图像的平均误差
	vector<Point2f> proImgPoints;
	for (int i=0; i< count; i++)
	{
		vector<Point3f> tempPointSet = objectPoints[i];
		fisheye::projectPoints(tempPointSet, proImgPoints, _rvec[i], _tvec[i], _K, _D);
		vector<Point2f> tempImgPoint = cornerSquare[i];
		Mat tempImgPointMat= Mat(1, tempImgPoint.size(), CV_32FC2);
		Mat proImgPointsMat = Mat(1, proImgPoints.size(), CV_32FC2);
		for (int j=0; j!=tempImgPoint.size(); j++)
		{
			proImgPointsMat.at<Vec2f>(0,j) = Vec2f (proImgPoints[j].x, proImgPoints[j].y);
			tempImgPointMat.at<Vec2f>(0,j)=Vec2f(tempImgPoint[j].x, tempImgPoint[j].y);
		}
		err=norm(proImgPointsMat, tempImgPointMat, NORM_L2);
		err /=pointCnts[i];
		total_err +=err;
		 fout<<"第"<<i+1<<"幅图像的平均误差："<<err<<"像素"<<endl;
	}
	fout<<"总体平均误差："<<total_err/count<<"像素"<<endl;
	cout<<"******评价完成******"<<endl;

	return 0;
}