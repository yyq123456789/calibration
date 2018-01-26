#include "opencv.hpp"
#include <fstream>
#include <iostream>

using namespace std;

#define CAMNUM  10

//ͨ�������������ڽǵ㹹�ɵ���������֮��ļн��жϽǵ�������
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

//����궨�����������ļ�
int writeParams(string fileName,  cv::Mat& matData)
{
	//���ļ�
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
    foutFile<<endl;    //ÿһ��Ϊһ������
    foutFile.close();
	return 0;
}

//������ͶӰ�����ų��������趨��ֵ�İб�ͼƬ
int evaluateCalibrationResult( vector<vector<cv::Point3f>> objectPoints, vector<vector<cv::Point2f>> cornerSquare, vector<int> pointCnts, vector<cv::Vec3d> _rvec,
	vector<cv::Vec3d> _tvec, cv::Mat _K, cv::Mat _D, int count, vector<int> &outLierIndex, int camCnt, int errThresh)
{
	stringstream ss;
	ss << camCnt;
	string evaluatPath = "data/result/evaluateCalibrationResult" + ss.str() + ".txt";
	ofstream fout(evaluatPath);

	double total_err = 0.0;//����ͼ���ƽ������
	double err=0.0;//����ͼ���ƽ�����
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
		fout<<"��"<<i<<"��ͼ��������ͶӰ��"<< maxValue <<"����"<<endl;

		//�ҳ���ͶӰ������2��ͼ
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

//��ʼ���ǵ����ά����
void init3DPoints(cv::Size boardSize, cv::Size squareSize, vector<cv::Point3f> &singlePatternPoint)
{
	for (int i = 0; i<boardSize.height; i++)
	{
		for (int j = 0; j<boardSize.width; j++)
		{
			cv::Point3f tempPoint;//�����ǵ����ά����
			tempPoint.x = i * squareSize.width;
			tempPoint.y = j * squareSize.height;
			tempPoint.z = 0;
			singlePatternPoint.push_back(tempPoint);
		}
	}
}
//��ȡ��Ƶ֡����ȡ�ǵ㣨�Ƚ��������ϲ�����
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
		cv::resize(grayImg, downSizeImg, cv::Size(), scale, scale, CV_INTER_LINEAR);//������ԭͼ
		//��ȡ�ǵ�
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
			//�ϲ���corner
			for (int num = 0; num < corners.size(); num++)
			{
				cv::Point2f tempPoint = corners[num];
				corners[num] = cv::Point2f(tempPoint.x / scale, tempPoint.y / scale);
			}

			//�����ؾ�ȷ��
			cornerSubPix(grayImg, corners, cv::Size(8, 8), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
			bool good = testCorners(corners, boardSize.width, boardSize.height);
			if (false==good)
			{
				continue;
			}
			//���Ƽ�⵽�Ľǵ㲢��ʾ
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

			successPatternNum++;//�ɹ���ȡ�ǵ������ͼ����
			cornersSeq.push_back(corners);//�����������̸�ǵ�
			imageSeq.push_back(image);//�������гɹ���ȡ���ǵ�����̸�ͼ��
		}
	}
	cap.release();

	return successPatternNum;
}



//������궨����
int main()
{
	string dirName = "H:/calibrationVideos/";
	cv::Size boardSize(8, 6); //�б�ͼƬ���ڽǵ����
	cv::Size squareSize(20, 20);//���̸�ߴ�
	vector<cv::Point3f> singlePatternPoints;//��������ͼ��Ľǵ���ά����
	float scale = 0.25;//�������߶�
	int interp = 3;  //��Ƶÿ��interp֡��ȡһ֡
	int errThresh = 2; //��ͶӰ�����ֵ
	//��ʼ�������б�ͼƬ����ά��
	init3DPoints(boardSize, squareSize, singlePatternPoints);

	//
	for (int camCnt = 0; camCnt < CAMNUM; camCnt++)
	{
		vector<cv::Point2f> corners;//�洢һ������ͼ�е����нǵ��ά����
		vector<vector<cv::Point2f>> cornersSeq;//�洢��������ͼ�ǵ�Ķ�ά����
		vector<cv::Mat> imageSeq;//�洢���гɹ���ȡ�ǵ������ͼ

		//��ȡ��Ƶ����ȡ�ǵ�
		stringstream dirNum;
		dirNum << camCnt;
		string videoPath = dirName + dirNum.str() + "_.mp4";
		cout << "********��ʼ��ȡ�ǵ㣡********" << endl;
		int successImgNum = videoFindCorner(videoPath, boardSize, corners, cornersSeq, imageSeq, scale, interp);
		cout << "********�ǵ���ȡ��ɣ�********" << endl;
		if (successImgNum < 5)
		{
			cout << "bad!" << endl;
			getchar();
			continue;
		}

		vector<vector<cv::Point3f>> objectPoints;//��������ͼ��Ľǵ���ά����
		vector<int> pointCounts;
		for (int n = 0; n<successImgNum; n++)
		{
			objectPoints.push_back(singlePatternPoints);
			pointCounts.push_back(boardSize.width * boardSize.height);
		}

		//�궨
		cout << "*****��ʼ�궨!******" << endl;
		cv::Size imageSize = imageSeq[0].size();
		cv::Mat intrinsicMatrix; //�ڲξ���
		cv::Mat disCoeffs;//����ϵ��
		vector<cv::Vec3d> rotVectors;//��ת����
		vector<cv::Vec3d> transVectors;//ƽ������
		int flags = 0;
		flags |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
		flags |= cv::fisheye::CALIB_CHECK_COND;
		flags |= cv::fisheye::CALIB_FIX_SKEW;
		cv::fisheye::calibrate(objectPoints, cornersSeq, imageSize, intrinsicMatrix, disCoeffs, rotVectors, transVectors, flags, cv::TermCriteria(3, 20, 1e-6));
		cout << "*****�궨��ɣ�*****" << endl;
		
		//����
		vector<int> outLierIndex;
		evaluateCalibrationResult(objectPoints, cornersSeq, pointCounts, rotVectors, transVectors, intrinsicMatrix, disCoeffs, successImgNum, outLierIndex, camCnt, errThresh);
		//ɾ������Ľǵ�ͼ
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
		//���±궨
		cv::fisheye::calibrate(newObjectPoints, newCornersSeq, imageSize, intrinsicMatrix, disCoeffs, rotVectors, transVectors, flags, cv::TermCriteria(3, 20, 1e-6));
		//���¼�����ͶӰ���
		outLierIndex.clear();
		evaluateCalibrationResult(objectPoints, newCornersSeq, pointCounts, rotVectors, transVectors, intrinsicMatrix, disCoeffs, successImgNum, outLierIndex, camCnt, errThresh);

		//����궨���
		string calibFile = "data/result/intrinsic_params" + dirNum.str() + ".txt";
		ofstream foutFile(calibFile, ios::trunc);
		foutFile.close();
		writeParams(calibFile, intrinsicMatrix);
		writeParams(calibFile, disCoeffs);

		//ͨ������У��Ч���鿴������궨Ч��
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