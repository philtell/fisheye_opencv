#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <ctime>
#include <cstdio>
#include <iomanip>
#include <filesystem>
// #include <experimental/filesystem> // Use this header for filesystem operations
#include <opencv2/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;
// namespace fs = std::experimental::filesystem; // Use the experimental::filesystem namespace
namespace fs = std::filesystem; // Use the std::filesystem namespace


int main()
{
    ofstream fout("caliberation_result.txt");  /**    保存定标结果的文件     **/

    /************************************************************************
    读取每一幅图像，从中提取出角点，然后对角点进行亚像素精确化
    *************************************************************************/
    string imageFolderPath = "/home/aicrobo/code/fisheye_opencv/image_path/images/"; // Change this to the folder containing your images
    cout << "开始提取角点………………" << endl;
    int image_count = 14;                    /****    图像数量     ****/
    Size board_size = Size(8, 11);            /****    定标板上每行、列的角点数       ****/
    vector<Point2f> corners;                  /****    缓存每幅图像上检测到的角点       ****/
    vector<vector<Point2f>>  corners_Seq;    /****  保存检测到的所有角点       ****/
    vector<Mat>  image_Seq;
    int successImageNum = 0;				/****	成功提取角点的棋盘图数量	****/

    int count = 0;
    int f = 0;
    for (const auto& entry : fs::directory_iterator(imageFolderPath))
    {
        string imageFilePath = entry.path().string();
        cout << "Processing: " << imageFilePath << endl;
        cv::Mat image = imread(imageFilePath);
        if (image.empty())
        {
            cout << "Failed to read image: " << imageFilePath << endl;
            continue;
        }
        /* 提取角点 */
        cv::Mat imageGray;
        cvtColor(image, imageGray, COLOR_BGR2GRAY);
        bool patternfound = findChessboardCorners(image, board_size, corners, (CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE) +
                                                                              CALIB_CB_FAST_CHECK);
        if (!patternfound)
        {
            cout << "can not find chessboard corners!\n";
            continue;
            exit(1);
        }
        else
        {
            /* 亚像素精确化 */
            cornerSubPix(imageGray, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
            /* 绘制检测到的角点并保存 */
            Mat imageTemp = image.clone();
            for (int j = 0; j < corners.size(); j++)
            {
                circle(imageTemp, corners[j], 10, Scalar(0, 0, 255), 2, 8, 0);
            }    std::stringstream ss;
            ss << f++;
            std::string str = ss.str();
            string imageFileName = "/home/aicrobo/code/fisheye_opencv/image_path/image_2/"+str;
            imageFileName += "_corner.jpg";
            imwrite(imageFileName, imageTemp);
            // cout << "Frame corner#" << i + 1 << "...end" << endl;

            count = count + corners.size();
            successImageNum = successImageNum + 1;
            corners_Seq.push_back(corners);
        }
        image_Seq.push_back(image);
    }
    cout << "角点提取完成！\n";
    /************************************************************************
    摄像机定标
    *************************************************************************/
    cout << "开始定标………………" << endl;
    Size square_size = Size(20, 20);
    vector<vector<Point3f>>  object_Points;        /****  保存定标板上角点的三维坐标   ****/

    Mat image_points = Mat(1, count, CV_32FC2, Scalar::all(0));  /*****   保存提取的所有角点   *****/
    vector<int>  point_counts;
    /* 初始化定标板上角点的三维坐标 */
    for (int t = 0; t<successImageNum; t++)
    {
        vector<Point3f> tempPointSet;
        for (int i = 0; i<board_size.height; i++)
        {
            for (int j = 0; j<board_size.width; j++)
            {
                /* 假设定标板放在世界坐标系中z=0的平面上 */
                Point3f tempPoint;
                tempPoint.x = i*square_size.width;
                tempPoint.y = j*square_size.height;
                tempPoint.z = 0;
                tempPointSet.push_back(tempPoint);
            }
        }
        object_Points.push_back(tempPointSet);
    }
    for (int i = 0; i< successImageNum; i++)
    {
        point_counts.push_back(board_size.width*board_size.height);
    }
    /* 开始定标 */
    Size image_size = image_Seq[0].size();
    cv::Matx33d intrinsic_matrix;    /*****    摄像机内参数矩阵    ****/
    cv::Vec4d distortion_coeffs;     /* 摄像机的4个畸变系数：k1,k2,k3,k4*/
    std::vector<cv::Vec3d> rotation_vectors;                           /* 每幅图像的旋转向量 */
    std::vector<cv::Vec3d> translation_vectors;                        /* 每幅图像的平移向量 */
    int flags = 0;
    flags |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
    flags |= cv::fisheye::CALIB_CHECK_COND;
    flags |= cv::fisheye::CALIB_FIX_SKEW;
    fisheye::calibrate(object_Points, corners_Seq, image_size, intrinsic_matrix, distortion_coeffs, rotation_vectors, translation_vectors, flags, cv::TermCriteria(3, 20, 1e-6));
    cout << "定标完成！\n";

    string datFileName = "/home/aicrobo/code/fisheye_opencv/dist_path/camParam.dat";
    FILE *camParam = fopen(datFileName.c_str(), "wb");
    if (camParam == NULL) {
        std::cout << "can not create data file: " << datFileName.c_str() << " !!!" << std::endl;
        return false;
    }
    fwrite(&intrinsic_matrix, sizeof(cv::Matx33d), 1, camParam);
    fwrite(&distortion_coeffs, sizeof(cv::Vec4d), 1, camParam);
    fwrite(&image_size, sizeof(Size), 1, camParam);
    fclose(camParam);


    /************************************************************************
    对定标结果进行评价
    *************************************************************************/
    cout << "开始评价定标结果………………" << endl;
    double total_err = 0.0;                   /* 所有图像的平均误差的总和 */
    double err = 0.0;                        /* 每幅图像的平均误差 */
    vector<Point2f>  image_points2;             /****   保存重新计算得到的投影点    ****/

    cout << "每幅图像的定标误差：" << endl;
    cout << "每幅图像的定标误差：" << endl << endl;
    for (int i = 0; i<image_count; i++)
    {
        vector<Point3f> tempPointSet = object_Points[i];
        /****    通过得到的摄像机内外参数，对空间的三维点进行重新投影计算，得到新的投影点     ****/
        fisheye::projectPoints(tempPointSet, image_points2, rotation_vectors[i], translation_vectors[i], intrinsic_matrix, distortion_coeffs);
        /* 计算新的投影点和旧的投影点之间的误差*/
        vector<Point2f> tempImagePoint = corners_Seq[i];
        Mat tempImagePointMat = Mat(1, tempImagePoint.size(), CV_32FC2);
        Mat image_points2Mat = Mat(1, image_points2.size(), CV_32FC2);
        for (size_t i = 0; i != tempImagePoint.size(); i++)
        {
            image_points2Mat.at<Vec2f>(0, i) = Vec2f(image_points2[i].x, image_points2[i].y);
            tempImagePointMat.at<Vec2f>(0, i) = Vec2f(tempImagePoint[i].x, tempImagePoint[i].y);
        }
        err = norm(image_points2Mat, tempImagePointMat, NORM_L2);
        total_err += err /= point_counts[i];
        cout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
        fout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
    }
    cout << "总体平均误差：" << total_err / image_count << "像素" << endl;
    fout << "总体平均误差：" << total_err / image_count << "像素" << endl << endl;
    cout << "评价完成！" << endl;

    /************************************************************************
    保存定标结果
    *************************************************************************/
    cout << "开始保存定标结果………………" << endl;
    Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); /* 保存每幅图像的旋转矩阵 */

    fout << "相机内参数矩阵：" << endl;
    fout << intrinsic_matrix << endl;
    fout << "畸变系数：\n";
    fout << distortion_coeffs << endl;
    for (int i = 0; i<image_count; i++)
    {
        fout << "第" << i + 1 << "幅图像的旋转向量：" << endl;
        fout << rotation_vectors[i] << endl;

        /* 将旋转向量转换为相对应的旋转矩阵 */
        Rodrigues(rotation_vectors[i], rotation_matrix);
        fout << "第" << i + 1 << "幅图像的旋转矩阵：" << endl;
        fout << rotation_matrix << endl;
        fout << "第" << i + 1 << "幅图像的平移向量：" << endl;
        fout << translation_vectors[i] << endl;
    }
    cout << "完成保存" << endl;
    fout << endl;


    /************************************************************************
    显示定标结果
    *************************************************************************/
    Mat mapx = Mat(image_size, CV_32FC1);
    Mat mapy = Mat(image_size, CV_32FC1);
    Mat R = Mat::eye(3, 3, CV_32F);
    cout << "保存矫正图像" << endl;
    for (int i = 0; i != image_count; i++)
    {
        cout << "Frame #" << i + 1 << "..." << endl;
        Mat newCameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));
        fisheye::initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, R, intrinsic_matrix, image_size, CV_32FC1, mapx, mapy);
        Mat t = image_Seq[i].clone();
        cv::remap(image_Seq[i], t, mapx, mapy, INTER_LINEAR);
        string imageFileName;
        std::stringstream StrStm;
        StrStm << i + 1;
        StrStm >> imageFileName;
        imageFileName += "_d.jpg";
        imwrite(imageFileName, t);
    }
    cout << "保存结束" << endl;


    /************************************************************************
    测试一张图片
    *************************************************************************/
    if (1)
    {
        Mat newCameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));
        Mat testImage = imread("/home/aicrobo/code/fisheye_opencv/image_path/images/img1.jpg", 1);

        cv::Matx33d intrinsic_matrix;
        cv::Vec4d distortion_coeffs;
        Size image_size;


        string datFileName = "/home/aicrobo/code/fisheye_opencv/dist_path/camParam.dat";

        FILE *camParam = fopen(datFileName.c_str(), "rb");
        if (camParam == NULL) {
            std::cout << "can not create data file: " << datFileName.c_str() << " !!!" << std::endl;
            return false;
        }
        fread(&intrinsic_matrix, sizeof(cv::Matx33d), 1, camParam);
        fread(&distortion_coeffs, sizeof(cv::Vec4d), 1, camParam);
        fread(&image_size, sizeof(Size), 1, camParam);
        fclose(camParam);


        Mat mapx = Mat(image_size, CV_32FC1);
        Mat mapy = Mat(image_size, CV_32FC1);
        Mat R = Mat::eye(3, 3, CV_32F);

        fisheye::initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, R, intrinsic_matrix, image_size, CV_32FC1, mapx, mapy);

        Mat t = testImage.clone();

        cv::remap(testImage, t, mapx, mapy, INTER_LINEAR);
        imwrite("TestOutput.jpg", t);
        cout << "保存结束" << endl;

        getchar();
    }


    return 0;
}