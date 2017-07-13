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
    //***************摄像机标定****************//
    double time0=getTickCount();
    vector<Point2f> corners;//存储一幅棋盘图中的所有角点二维坐标
    vector<vector<Point2f>> cornersSeq;//存储所有棋盘图角点的二维坐标
    vector<Mat> image_Seq;//存储所有棋盘图
    int successImgNum=0;
    int count=0;
    cout<<"********开始提取角点！********"<<endl;
    Mat image;
    for (int i=0; i<imgNum; i++)
    {
	    cout<<"Image#"<<i<<"......."<<endl;
        image=patternImgList[i].clone();
	    /**********************提取角点*************************/
	    bool patternfound= findChessboardCorners(image, boardSize,
            corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE+CALIB_CB_FAST_CHECK);
	    if (!patternfound)
	    {
		    cout<<"Can not find chess board corners!\n"<<endl;
		    continue;
	    }
	    else
	    {
		    /************************亚像素精确化******************************/
		    cornerSubPix(image, corners, Size(11, 11), Size(-1,-1) , TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
		    /************************绘制检测到的角点并显示******************************/
		    Mat cornerImg = image.clone();
            cvtColor(cornerImg, cornerImg, CV_GRAY2BGR);
		    for (int j=0; j< corners.size(); j++)
		    {
			    circle(cornerImg, corners[j], 10, Scalar(0,0,255), 2, 8, 0);
		    }
            namedWindow("CirclePattern");
            imshow("CirclePattern", cornerImg);
            waitKey(1);

		    count +=corners.size();//所有棋盘图中的角点个数
		    successImgNum++;//成功提取角点的棋盘图个数
		    cornersSeq.push_back(corners);
	    }
	    image_Seq.push_back(image);
    }
    cout<<"*******角点提取完成！******"<<endl;
	
    /**************************摄像机标定******************************/
    Size squre_size=Size(20,20);//棋盘格尺寸
    vector<vector<Point3f>> object_points;//所有棋盘图像的角点三维坐标
    vector<int> pointCounts;
    /*初始化标定板上的三维坐标*/
    for(int n=0; n<successImgNum; n++)
    {
	    vector<Point3f> tempPointSet;//单幅棋盘图像的所有角点
	    for (int i=0; i<boardSize.height; i++)
	    {
		    for (int j=0; j<boardSize.width; j++)
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
	    pointCounts.push_back(boardSize.width * boardSize.height);//保存每一幅棋盘图的角点个数
    }
    /***开始标定***/
    cout<<"*****开始标定!******"<<endl;
    Size imgSize=Size(imgWidth, imgHeight);
    vector<Vec3d> rotation;//旋转向量
    vector<Vec3d> translation;//平移向量
    int flags=0;
    flags |= fisheye::CALIB_RECOMPUTE_EXTRINSIC;
    flags |= fisheye::CALIB_CHECK_COND;
    flags |= fisheye::CALIB_FIX_SKEW;
    fisheye::calibrate(object_points, cornersSeq, imgSize, camK, camDiscoeff, rotation, translation, flags, cv::TermCriteria(3, 20, 1e-6));
    cout<<"*****标定完成！*****"<<endl;
    double time1=getTickCount();
    cout<<"Calibration time :"<<(time1-time0)/getTickFrequency()<<"s"<<endl;
    evaluateCalibrationResult( object_points, cornersSeq, pointCounts,  rotation,  translation, camK, camDiscoeff, successImgNum);
}

bool CCalibration::evaluateCalibrationResult( vector<vector<Point3f>> objectPoints, vector<vector<Point2f>> cornerSquare, vector<int> pointCnts,  vector<Vec3d> _rvec,  vector<Vec3d> _tvec, Mat _K, Mat _D, int count)
{
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
		cout<<"第"<<i+1<<"幅图像的平均误差："<<err<<"像素"<<endl;
	}
	cout<<"总体平均误差："<<total_err/count<<"像素"<<endl;
	cout<<"******评价完成******"<<endl;

	return true;
}

void CCalibration::run()
{
    bool readSuccess=readPatternImg();
    calibProcess();
    writeParams();
}