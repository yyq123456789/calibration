#include "opencv.hpp"
#include "calibration.h"
#include <fstream>
#include <iostream>

using namespace std;
using namespace cv;

#define ImgNum  11

 //������궨

int main()
{
//***************������궨****************//
    double time0=getTickCount();
    string calibFile="Results/calibResult.txt";
    ofstream fileout(calibFile, ios::trunc);//���ԭ������
    fileout.close();
    int imgCount = ImgNum;//���̸�ͼƬ����
    Size board_size = Size(9,6);//�ǵ����
    vector<Point2f> corners;//�洢һ������ͼ�е����нǵ��ά����
    vector<vector<Point2f>> corners_Seq;//�洢��������ͼ�ǵ�Ķ�ά����
    vector<Mat> image_Seq;//�洢��������ͼ
    int successImgNum=0;
    int count=0;
    cout<<"********��ʼ��ȡ�ǵ㣡********"<<endl;
    for (int i=0; i<imgCount; i++)
    {
	    cout<<"Image#"<<i+1<<"......."<<endl;
	    string imgFileName;
	    stringstream StrStm;
	    string filenum;
	    StrStm<<i+1;
	    StrStm>>filenum;
	    imgFileName ="./Images/Pattern/img" +filenum+ ".jpg";
	    Mat image = imread(imgFileName);   //�����ļ������ζ�ȡ����ͼ
        pyrDown(image, image);
        pyrDown(image, image);
	    /**********************��ȡ�ǵ�*************************/
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
		    /************************�����ؾ�ȷ��******************************/
		    cornerSubPix(greyImg, corners, Size(11, 11), Size(-1,-1) , TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
		    /************************���Ƽ�⵽�Ľǵ㲢����******************************/
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
		    imwrite(tempFileName, tempImg);//���ǵ���Ƴ��󱣴�
		    cout<<"Corner_Image#"<<i+1<<"......"<<endl;

		    count +=corners.size();//��������ͼ�еĽǵ����
		    successImgNum++;//�ɹ���ȡ�ǵ������ͼ����
		    corners_Seq.push_back(corners);
	    }
	    image_Seq.push_back(image);
    }
    cout<<"*******�ǵ���ȡ��ɣ�******"<<endl;
	
    /**************************������궨******************************/
    Size squre_size=Size(20,20);//���̸�ߴ�
    vector<vector<Point3f>> object_points;//��������ͼ��Ľǵ���ά����
    vector<int> point_counts;
    /*��ʼ���궨���ϵ���ά����*/
    for(int n=0; n<successImgNum; n++)
    {
	    vector<Point3f> tempPointSet;//��������ͼ������нǵ�
	    for (int i=0; i<board_size.height; i++)
	    {
		    for (int j=0; j<board_size.width; j++)
		    {
			    Point3f tempPoint;//�����ǵ����ά����
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
	    point_counts.push_back(board_size.width * board_size.height);//����ÿһ������ͼ�Ľǵ����
    }
    /***��ʼ�궨***/
    cout<<"*****��ʼ�궨!******"<<endl;
    Size image_size=image_Seq[0].size();
    Mat intrinsic_matrix; //�ڲξ���
    Mat distortion_coeffs;//����ϵ��
    vector<Vec3d> rotation_vectors;//��ת����
    vector<Vec3d> translation_vectors;//ƽ������
    int flags=0;
    flags |= fisheye::CALIB_RECOMPUTE_EXTRINSIC;
    flags |= fisheye::CALIB_CHECK_COND;
    flags |= fisheye::CALIB_FIX_SKEW;
    fisheye::calibrate(object_points, corners_Seq, image_size, intrinsic_matrix, distortion_coeffs, rotation_vectors, translation_vectors, flags, cv::TermCriteria(3, 20, 1e-6));
    cout<<"*****�궨��ɣ�*****"<<endl;
    double time1=getTickCount();
    cout<<"Calibration time :"<<(time1-time0)/getTickFrequency()<<"s"<<endl;

    /****************************�Ա궨�����������*****************************/
    evaluateCalibrationResult( object_points, corners_Seq, point_counts,  rotation_vectors,  translation_vectors, intrinsic_matrix, distortion_coeffs, imgCount);

    /*********************����궨���******************/
	    cout<<"*****��ʼ����궨���*****"<<endl; 
    //writeParams(intrinsicFile, intrinsic_matrix );//�����ڲξ���
    //writeParams(disCoeffsFile, distortion_coeffs );//�������ϵ��
    writeParams( calibFile, intrinsic_matrix );
    writeParams( calibFile, distortion_coeffs );
    cout<<"******��ɱ���******"<<endl;
      
}

/******************************************************************************
 * ��������:    writeParams()
 * ��������:    ���궨��ȡ�Ĳ������ڲκͻ���ϵ���������ָ��txt�ļ�
 * ��    ��:        matData               -��������
 * ��    ��:        fileName              -�ļ���
 * �� �� ֵ:       0                         -�ɹ�
                        ����                    -ʧ��
 ******************************************************************************/
int writeParams(string fileName,  cv::Mat& matData)
{
	int retVal=0;

	//���ļ�
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
    foutFile<<endl;    //ÿһ��Ϊһ������
    foutFile.close();
	return retVal;
}
/******************************************************************************
 * ��������:    evaluateCalibrationResult()
 * ��������:    ���۱궨������������۽��
 * ��    ��:        objectPoints               -��ά�ǵ�����
                        cornerSquare             -��ά�ǵ�����
                        pointCnts                   -�ǵ����
                        _rvec                        -��ת����
                        _tvec                        -ƽ������
                        _K                            -�ڲξ���
                        _D                            -����ϵ��
                        count                         -���̸�ͼƬ����
 * ��    ��:        ��
 * �� �� ֵ:       0                              -�ɹ�
                        ����                         -ʧ��
 ******************************************************************************/
int evaluateCalibrationResult( vector<vector<Point3f>> objectPoints, vector<vector<Point2f>> cornerSquare, vector<int> pointCnts,  vector<Vec3d> _rvec,  vector<Vec3d> _tvec, Mat _K, Mat _D, int count)
{
	ofstream fout("./Results/evaluateCalibrationResult.txt");
	cout<<"******��ʼ���۱궨���******"<<endl;
	double total_err = 0.0;//����ͼ���ƽ������
	double err=0.0;//����ͼ���ƽ�����
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
		 fout<<"��"<<i+1<<"��ͼ���ƽ����"<<err<<"����"<<endl;
	}
	fout<<"����ƽ����"<<total_err/count<<"����"<<endl;
	cout<<"******�������******"<<endl;

	return 0;
}