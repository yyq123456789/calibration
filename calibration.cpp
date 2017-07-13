#include "calibration.h"


bool CCalibration::writeParams()
{
    camK.convertTo(camK, CV_32FC1);
    camDiscoeff.convertTo(camDiscoeff, CV_32FC1);
    ofstream out;
    out.open(calibResultPath+"calibResult.txt", ios::out);
    out<<camK.at<float>(0, 0)<<endl;
    out<<camK.at<float>(1, 1)<<endl;
    out<<camK.at<float>(0, 2)<<endl;
    out<<camK.at<float>(1, 2)<<endl;

    out<<camDiscoeff.at<float>(0,0)<<endl;
    out<<camDiscoeff.at<float>(1,0)<<endl;
    out<<camDiscoeff.at<float>(2,0)<<endl;
    out<<camDiscoeff.at<float>(3,0)<<endl;
    out.close();
    return true;
}

bool CCalibration::readPatternImg()
{
    int imgNum=0;
    Mat img;
    do
    {
        stringstream ss;
        ss<<imgNum;
        string path=patternImgPath+ss.str()+".jpg";
        img=imread(path, 0);
        if (!img.data)
        {
            break;
        }
        pyrDown(img, img);
        pyrDown(img, img);
        patternImgList.push_back(img);
        imgNum++;
    } while(true);
    if (imgNum==0)
    {
        cout<<" error! No pattern imgs!"<<endl;
        return false;
    }
    this->imgNum=imgNum;
    imgHeight=patternImgList[0].rows;
    imgWidth=patternImgList[0].cols;

    return true;
}

void CCalibration::calibProcess()
{
    //***************������궨****************//
    double time0=getTickCount();
    vector<Point2f> corners;//�洢һ������ͼ�е����нǵ��ά����
    vector<vector<Point2f>> cornersSeq;//�洢��������ͼ�ǵ�Ķ�ά����
    vector<Mat> image_Seq;//�洢��������ͼ
    int successImgNum=0;
    int count=0;
    cout<<"********��ʼ��ȡ�ǵ㣡********"<<endl;
    Mat image;
    for (int i=0; i<imgNum; i++)
    {
	    cout<<"Image#"<<i<<"......."<<endl;
        image=patternImgList[i].clone();
	    /**********************��ȡ�ǵ�*************************/
	    bool patternfound= findChessboardCorners(image, boardSize,
            corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE+CALIB_CB_FAST_CHECK);
	    if (!patternfound)
	    {
		    cout<<"Can not find chess board corners!\n"<<endl;
		    continue;
	    }
	    else
	    {
		    /************************�����ؾ�ȷ��******************************/
		    cornerSubPix(image, corners, Size(11, 11), Size(-1,-1) , TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
		    /************************���Ƽ�⵽�Ľǵ㲢��ʾ******************************/
		    Mat cornerImg = image.clone();
            cvtColor(cornerImg, cornerImg, CV_GRAY2BGR);
		    for (int j=0; j< corners.size(); j++)
		    {
			    circle(cornerImg, corners[j], 10, Scalar(0,0,255), 2, 8, 0);
		    }
            namedWindow("CirclePattern");
            imshow("CirclePattern", cornerImg);
            waitKey(1);

		    count +=corners.size();//��������ͼ�еĽǵ����
		    successImgNum++;//�ɹ���ȡ�ǵ������ͼ����
		    cornersSeq.push_back(corners);
	    }
	    image_Seq.push_back(image);
    }
    cout<<"*******�ǵ���ȡ��ɣ�******"<<endl;
	
    /**************************������궨******************************/
    Size squre_size=Size(20,20);//���̸�ߴ�
    vector<vector<Point3f>> object_points;//��������ͼ��Ľǵ���ά����
    vector<int> pointCounts;
    /*��ʼ���궨���ϵ���ά����*/
    for(int n=0; n<successImgNum; n++)
    {
	    vector<Point3f> tempPointSet;//��������ͼ������нǵ�
	    for (int i=0; i<boardSize.height; i++)
	    {
		    for (int j=0; j<boardSize.width; j++)
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
	    pointCounts.push_back(boardSize.width * boardSize.height);//����ÿһ������ͼ�Ľǵ����
    }
    /***��ʼ�궨***/
    cout<<"*****��ʼ�궨!******"<<endl;
    Size imgSize=Size(imgWidth, imgHeight);
    vector<Vec3d> rotation;//��ת����
    vector<Vec3d> translation;//ƽ������
    int flags=0;
    flags |= fisheye::CALIB_RECOMPUTE_EXTRINSIC;
    flags |= fisheye::CALIB_CHECK_COND;
    flags |= fisheye::CALIB_FIX_SKEW;
    fisheye::calibrate(object_points, cornersSeq, imgSize, camK, camDiscoeff, rotation, translation, flags, cv::TermCriteria(3, 20, 1e-6));
    cout<<"*****�궨��ɣ�*****"<<endl;
    double time1=getTickCount();
    cout<<"Calibration time :"<<(time1-time0)/getTickFrequency()<<"s"<<endl;
    evaluateCalibrationResult( object_points, cornersSeq, pointCounts,  rotation,  translation, camK, camDiscoeff, successImgNum);
}

bool CCalibration::evaluateCalibrationResult( vector<vector<Point3f>> objectPoints, vector<vector<Point2f>> cornerSquare, vector<int> pointCnts,  vector<Vec3d> _rvec,  vector<Vec3d> _tvec, Mat _K, Mat _D, int count)
{
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
		cout<<"��"<<i+1<<"��ͼ���ƽ����"<<err<<"����"<<endl;
	}
	cout<<"����ƽ����"<<total_err/count<<"����"<<endl;
	cout<<"******�������******"<<endl;

	return true;
}

void CCalibration::run()
{
    bool readSuccess=readPatternImg();
    calibProcess();
    writeParams();
}