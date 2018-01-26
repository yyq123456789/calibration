#include "opencv.hpp"
#include <fstream>
#include <iostream>

using namespace std;

#define CAMNUM  10

//通过计算三个相邻角点构成的两个向量之间的夹角判断角点连接性
bool testCorners(vector<cv::Point2f>& corners, int patternWidth, int patternHeight)
{
	if (corners.size() != patternWidth * patternHeight)
	{
		return false;
	}
	double dx1, dx2, dy1, dy2;
	double cosVal;
	for (int i = 0; i < patternHeight; ++i)
	{
		for (int j = 0; j < patternWidth - 2; ++j)
		{
			dx1 = corners[i*patternWidth + j + 1].x - corners[i*patternWidth + j].x;
			dy1 = corners[i*patternWidth + j + 1].y - corners[i*patternWidth + j].y;
			dx2 = corners[i*patternWidth + j + 2].x - corners[i*patternWidth + j + 1].x;
			dy2 = corners[i*patternWidth + j + 2].y - corners[i*patternWidth + j + 1].y;
			cosVal = (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2));
			if (fabs(cosVal) < 0.993)
			{
				return false;
			}
		}
	}
	for (int i = 0; i < patternHeight - 2; ++i)
	{
		for (int j = 0; j < patternWidth; ++j)
		{
			dx1 = corners[(i + 1)*patternWidth + j].x - corners[i*patternWidth + j].x;
			dy1 = corners[(i + 1)*patternWidth + j].y - corners[i*patternWidth + j].y;
			dx2 = corners[(i + 2)*patternWidth + j].x - corners[(i+1)*patternWidth + j].x;
			dy2 = corners[(i + 2)*patternWidth + j].y - corners[(i+1)*patternWidth + j].y;
			cosVal = (dx1 * dx2 + dy1 * dy2) / sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2));
			if (fabs(cosVal) < 0.993)
			{
				return false;
			}
		}
	}
	return true;
}

void genNewCamK(cv::Size imgSize, cv::Mat &newCamK)
{
	float FOVX = 120 * M_PI / 180;
	newCamK = cv::Mat::eye(3, 3, CV_32F);
	newCamK.at<float>(0, 2) = imgSize.width / 2;
	newCamK.at<float>(1, 2) = imgSize.height / 2;
	newCamK.at<float>(0, 0) = newCamK.at<float>(0, 2) / tan(FOVX / 2);
	newCamK.at<float>(1, 1) = newCamK.at<float>(0, 0);
	newCamK.at<float>(2, 2) = 1;
}

//保存标定参数到本地文件
int writeParams(string fileName,  cv::Mat& matData)
{
	//打开文件
	ofstream foutFile(fileName, ios::app);
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
	return 0;
}

//估计重投影误差，并排除误差大于设定阈值的靶标图片
int evaluateCalibrationResult( vector<vector<cv::Point3f>> objectPoints, vector<vector<cv::Point2f>> cornerSquare, vector<int> pointCnts, vector<cv::Vec3d> _rvec,
	vector<cv::Vec3d> _tvec, cv::Mat _K, cv::Mat _D, int count, vector<int> &outLierIndex, int camCnt, int errThresh)
{
	stringstream ss;
	ss << camCnt;
	string evaluatPath = "data/result/evaluateCalibrationResult" + ss.str() + ".txt";
	ofstream fout(evaluatPath);

	double total_err = 0.0;//所有图像的平均误差和
	double err=0.0;//单幅图像的平均误差
	vector<cv::Point2f> proImgPoints;
	for (int i=0; i< count; i++)
	{
		float maxValue = -1;
		vector<cv::Point3f> tempPointSet = objectPoints[i];
		cv::fisheye::projectPoints(tempPointSet, proImgPoints, _rvec[i], _tvec[i], _K, _D);
		vector<cv::Point2f> tempImgPoint = cornerSquare[i];
		cv::Mat tempImgPointMat= cv::Mat(1, tempImgPoint.size(), CV_32FC2);
		cv::Mat proImgPointsMat = cv::Mat(1, proImgPoints.size(), CV_32FC2);
		for (int j=0; j!=tempImgPoint.size(); j++)
		{
			proImgPointsMat.at<cv::Vec2f>(0,j) = cv::Vec2f (proImgPoints[j].x, proImgPoints[j].y);
			tempImgPointMat.at<cv::Vec2f>(0, j) = cv::Vec2f(tempImgPoint[j].x, tempImgPoint[j].y);
			float dx = proImgPoints[j].x - tempImgPoint[j].x;
			float dy = proImgPoints[j].y - tempImgPoint[j].y;
			float diff = sqrt(dx*dx+dy*dy);
			if (diff > maxValue)
			{
				maxValue = diff;
			}
		}	
		fout<<"第"<<i<<"幅图像的最大重投影误差："<< maxValue <<"像素"<<endl;

		//找出重投影误差大于2的图
		if (maxValue > errThresh)
		{
			outLierIndex.push_back(-1);
		}
		else
		{
			outLierIndex.push_back(0);
		}
	}
	fout.close();
	return 0;
}

//初始化角点的三维坐标
void init3DPoints(cv::Size boardSize, cv::Size squareSize, vector<cv::Point3f> &singlePatternPoint)
{
	for (int i = 0; i<boardSize.height; i++)
	{
		for (int j = 0; j<boardSize.width; j++)
		{
			cv::Point3f tempPoint;//单个角点的三维坐标
			tempPoint.x = i * squareSize.width;
			tempPoint.y = j * squareSize.height;
			tempPoint.z = 0;
			singlePatternPoint.push_back(tempPoint);
		}
	}
}
//读取视频帧并提取角点（先降采样后上采样）
int videoFindCorner(string path,cv::Size boardSize, vector<cv::Point2f> &corners, vector<vector<cv::Point2f>> &cornersSeq, vector<cv::Mat> &imageSeq,float scale, int interp)
{
	int successPatternNum = 0;
	cv::VideoCapture cap(path);
	int frameCnt = cap.get(cv::CAP_PROP_FRAME_COUNT);
	for (int i = 0; i<frameCnt; i++)
	{
		cv::Mat image, downSizeImg;
		bool flag = false;
		cout << "Image#" << i << "......." << endl;
		for (int j = 0; j < interp; j++)
		{
			cap.read(image);
			if (!cap.read(image))
			{
				flag = true;
				break;
			}
		}
		if (flag == true) break;

		cv::Mat grayImg;
		cvtColor(image, grayImg, CV_BGR2GRAY);
		cv::resize(grayImg, downSizeImg, cv::Size(), scale, scale, CV_INTER_LINEAR);//降采样原图
		//提取角点
		int calibFlag = cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK;
		bool patternfound = findChessboardCorners(downSizeImg, boardSize, corners, calibFlag);
		if (false==patternfound)
		{
			cout << "Can not find chess board corners!\n" << endl;
			continue;
		}
		else
		{
			cout << "find corners in Corner_Image#" << i << "......" << endl;
			//上采样corner
			for (int num = 0; num < corners.size(); num++)
			{
				cv::Point2f tempPoint = corners[num];
				corners[num] = cv::Point2f(tempPoint.x / scale, tempPoint.y / scale);
			}

			//亚像素精确化
			cornerSubPix(grayImg, corners, cv::Size(8, 8), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
			bool good = testCorners(corners, boardSize.width, boardSize.height);
			if (false==good)
			{
				continue;
			}
			//绘制检测到的角点并显示
			cv::Mat tempImg = image.clone();
			for (int j = 0; j< corners.size(); j++)
			{
				circle(tempImg, corners[j], 4, cv::Scalar(0, 0, 255), 2, 8, 0);
			}
			string winName = "Corners";
			cv::namedWindow(winName, 0);
			cv::resizeWindow(winName, 1200, 900);
			cv::imshow(winName, tempImg);
			cv::waitKey(1);

			successPatternNum++;//成功提取角点的棋盘图个数
			cornersSeq.push_back(corners);//保存所有棋盘格角点
			imageSeq.push_back(image);//保存所有成功提取到角点的棋盘格图像
		}
	}
	cap.release();

	return successPatternNum;
}



//摄像机标定代码
int main()
{
	string dirName = "H:/calibrationVideos/";
	cv::Size boardSize(8, 6); //靶标图片中内角点个数
	cv::Size squareSize(20, 20);//棋盘格尺寸
	vector<cv::Point3f> singlePatternPoints;//所有棋盘图像的角点三维坐标
	float scale = 0.25;//降采样尺度
	int interp = 3;  //视频每隔interp帧读取一帧
	int errThresh = 2; //重投影误差阈值
	//初始化单幅靶标图片的三维点
	init3DPoints(boardSize, squareSize, singlePatternPoints);

	//
	for (int camCnt = 0; camCnt < CAMNUM; camCnt++)
	{
		vector<cv::Point2f> corners;//存储一幅棋盘图中的所有角点二维坐标
		vector<vector<cv::Point2f>> cornersSeq;//存储所有棋盘图角点的二维坐标
		vector<cv::Mat> imageSeq;//存储所有成功提取角点的棋盘图

		//读取视频并提取角点
		stringstream dirNum;
		dirNum << camCnt;
		string videoPath = dirName + dirNum.str() + "_.mp4";
		cout << "********开始提取角点！********" << endl;
		int successImgNum = videoFindCorner(videoPath, boardSize, corners, cornersSeq, imageSeq, scale, interp);
		cout << "********角点提取完成！********" << endl;
		if (successImgNum < 5)
		{
			cout << "bad!" << endl;
			getchar();
			continue;
		}

		vector<vector<cv::Point3f>> objectPoints;//所有棋盘图像的角点三维坐标
		vector<int> pointCounts;
		for (int n = 0; n<successImgNum; n++)
		{
			objectPoints.push_back(singlePatternPoints);
			pointCounts.push_back(boardSize.width * boardSize.height);
		}

		//标定
		cout << "*****开始标定!******" << endl;
		cv::Size imageSize = imageSeq[0].size();
		cv::Mat intrinsicMatrix; //内参矩阵
		cv::Mat disCoeffs;//畸变系数
		vector<cv::Vec3d> rotVectors;//旋转向量
		vector<cv::Vec3d> transVectors;//平移向量
		int flags = 0;
		flags |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
		flags |= cv::fisheye::CALIB_CHECK_COND;
		flags |= cv::fisheye::CALIB_FIX_SKEW;
		cv::fisheye::calibrate(objectPoints, cornersSeq, imageSize, intrinsicMatrix, disCoeffs, rotVectors, transVectors, flags, cv::TermCriteria(3, 20, 1e-6));
		cout << "*****标定完成！*****" << endl;
		
		//评价
		vector<int> outLierIndex;
		evaluateCalibrationResult(objectPoints, cornersSeq, pointCounts, rotVectors, transVectors, intrinsicMatrix, disCoeffs, successImgNum, outLierIndex, camCnt, errThresh);
		//删除误差大的角点图
		vector<vector<cv::Point2f>> newCornersSeq;
		successImgNum = 0;
		for (int i = 0; i < cornersSeq.size(); i++)
		{
			if (outLierIndex[i] == 0)
			{
				newCornersSeq.push_back(cornersSeq[i]);
				successImgNum++;
			}
		}
		vector<vector<cv::Point3f>> newObjectPoints;
		for (int n = 0; n<successImgNum; n++)
		{
			newObjectPoints.push_back(singlePatternPoints);
		}
		//重新标定
		cv::fisheye::calibrate(newObjectPoints, newCornersSeq, imageSize, intrinsicMatrix, disCoeffs, rotVectors, transVectors, flags, cv::TermCriteria(3, 20, 1e-6));
		//重新计算重投影误差
		outLierIndex.clear();
		evaluateCalibrationResult(objectPoints, newCornersSeq, pointCounts, rotVectors, transVectors, intrinsicMatrix, disCoeffs, successImgNum, outLierIndex, camCnt, errThresh);

		//保存标定结果
		string calibFile = "data/result/intrinsic_params" + dirNum.str() + ".txt";
		ofstream foutFile(calibFile, ios::trunc);
		foutFile.close();
		writeParams(calibFile, intrinsicMatrix);
		writeParams(calibFile, disCoeffs);

		//通过畸变校正效果查看摄像机标定效果
		cv::Mat R = cv::Mat::eye(3, 3, CV_32FC1);
		cv::Mat mapx, mapy, newCamK, undistortImg, showImg;
		//cv::fisheye::estimateNewCameraMatrixForUndistortRectify(intrinsic_matrix, distortion_coeffs, image_size, R, newCamK, 0, image_size, 1);
		cv::fisheye::initUndistortRectifyMap(intrinsicMatrix, disCoeffs, R, intrinsicMatrix, imageSize, CV_32FC1, mapx, mapy);
		cv::remap(imageSeq[0], undistortImg, mapx, mapy, CV_INTER_LINEAR);
		cv::resize(undistortImg, showImg, cv::Size(), 0.25, 0.25, CV_INTER_LINEAR);
		string winName = "undistortImg";
		cv::namedWindow(winName, 1);
		cv::imshow(winName, showImg);
		cv::waitKey(0);
	}
}