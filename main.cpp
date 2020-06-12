#include <iostream>
//#include <filesystem>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <utility>
#include <cassert>
#include <vector>
//#include <opencv2/core/ocl.hpp>
#include "helping_functions/config_parser.h"



inline std::chrono::high_resolution_clock::time_point get_current_time_fenced()
{
    std::atomic_thread_fence(std::memory_order_seq_cst);
    auto res_time = std::chrono::high_resolution_clock::now();
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return res_time;
}


template<class D>
inline long long to_us(const D& d)
{
    return std::chrono::duration_cast<std::chrono::microseconds>(d).count();
}



void process_images(cv::Size &chessboard_size, std::vector<std::vector<cv::Point2f>>& allFoundCorners) {
//    Function to iterate through images
//    taken for camera calibration
    std::vector<cv::String> filenames;
    std::string path_to_directory = "../calibration_images";
    cv::glob(path_to_directory, filenames);
    std::vector<cv::Point2f> pointBuffer;
    for (const auto & filename : filenames) {
        if (filename != "../calibration_images/.DS_Store"){
            cv::Mat im = cv::imread(filename);
            bool found = findChessboardCorners(im, chessboard_size, pointBuffer,
                                               cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
            if (!found) {
                std::cout << filename << std::endl;
            }
            if (found) {
                cv::Mat gray_im;
                cvtColor(im, gray_im, cv::COLOR_BGR2GRAY);
                cv::TermCriteria criteria = cv::TermCriteria( cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                                                      30,0.1 );
                cv::Size winSize = cv::Size( 11, 11);
                cv::Size zeroZone = cv::Size( -1, -1 );
                //cornerSubPix is the algorithm focused on relocating the points. it receives the image, the corners
                // a window size, zeroZone and the actual criteria. The window size is the search area.
                cornerSubPix(gray_im, pointBuffer, winSize, zeroZone, criteria );
                allFoundCorners.push_back(pointBuffer);
            }
        }
    }
}


void calibration_process(cv::Mat& cameraMatrix, cv::Mat& distortionCoefficients) {
    //
    cv::Size chessboardDimensions = cv::Size(6, 9);
    float calibrationSquareDimension = 0.029f;

    std::vector <std::vector<cv::Point2f>> checkerboardImageSpacePoints;
    process_images(chessboardDimensions, checkerboardImageSpacePoints);

    //
    std::vector <std::vector<cv::Point3f>> worldSpaceCornerPoints(1);
    for (int i = 0; i < chessboardDimensions.height; i++) {
        for (int j = 0; j < chessboardDimensions.width; j++) {
            worldSpaceCornerPoints[0].emplace_back((float)j * calibrationSquareDimension, (float)i * calibrationSquareDimension, 0.0f);
        }
    }
    worldSpaceCornerPoints.resize(checkerboardImageSpacePoints.size(), worldSpaceCornerPoints[0]);


    //rotation and translation vectors (rVectors, tVectors)
    std::vector <cv::Mat> rVectors, tVectors;
//    distortionCoefficients = cv::Mat::zeros(1, 5, CV_64F);;
    double res = calibrateCamera(worldSpaceCornerPoints, checkerboardImageSpacePoints, chessboardDimensions, cameraMatrix,
                                 distortionCoefficients, rVectors, tVectors,
                                 (cv::CALIB_FIX_K4 | cv::CALIB_FIX_K5) + cv::CALIB_FIX_INTRINSIC);
    std::cout << "Reprojection Error (from calibrateCamera): " << res << std::endl;
}


void undistort(cv::Mat& cameraMatrix, cv::Mat& distortionCoefficients) {
//    UMat Uimg, Udst;
//    Uimg = imread(name.c_str(), IMREAD_UNCHANGED).getUMat(ACCESS_READ);
//    cv::UMat im1 = cv::imread("../working_images/left.jpg", cv::IMREAD_UNCHANGED).getUMat(cv::ACCESS_READ);
//    cv::UMat im2 = cv::imread("../working_images/right.jpg", cv::IMREAD_UNCHANGED).getUMat(cv::ACCESS_READ);
    cv::Mat im1 = cv::imread("../working_images/left.jpg");
    cv::Mat im2 = cv::imread("../working_images/right.jpg");
    if (im1.size().width != im2.size().width || im1.size().height != im2.size().height){
        std::cerr << "Images size does not match!"<< std::endl;
    }
    int width = im1.size().width;
    int height = im1.size().height;
    cv::Mat new_camera_matrix = getOptimalNewCameraMatrix(cameraMatrix, distortionCoefficients,
                                                      {width, height}, 1, {width, height}, nullptr);
    cv::Mat im1_udist, im2_undist;
    undistort(im1, im1_udist, cameraMatrix, distortionCoefficients, new_camera_matrix);
    undistort(im2, im2_undist, cameraMatrix, distortionCoefficients, new_camera_matrix);
    imwrite("../working_images/undistorted_left.jpg", im1);
    imwrite("../working_images/undistorted_right.jpg", im2);
}


void disparity(Configuration& configuration) {
    cv::Mat im1 = cv::imread("../working_images/undistorted_left.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat im2 = cv::imread("../working_images/undistorted_right.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat filtered_disp, conf_map;
    conf_map = cv::Mat(im1.rows,im2.cols,CV_8U);
    conf_map = cv::Scalar(255);

    double lambda = configuration.lambda;
    double sigma = configuration.sigma;
    double vis_mult = configuration.vis_mult;
    int preFilterCap = configuration.preFilterCap, disparityRange = configuration.disparityRange,
    minDisparity = configuration.minDisparity, uniquenessRatio = configuration.uniquenessRatio,
    windowSize = configuration.windowSize, smoothP1 = configuration.smoothP1 * windowSize * windowSize,
    smoothP2 = configuration.smoothP2 * windowSize * windowSize, disparityMaxDiff = configuration.disparityMaxDiff,
    speckleRange = configuration.speckleRange, speckleWindowSize = configuration.speckleWindowSize;

    bool mode = cv::StereoSGBM::MODE_SGBM_3WAY;
    cv::Mat left_disparity, right_disparity ,norm_disparity;
    cv::Ptr<cv::StereoSGBM> left_matcher = cv::StereoSGBM::create(minDisparity,
            disparityRange * 16, windowSize, smoothP1, smoothP2, disparityMaxDiff, preFilterCap,
            uniquenessRatio, speckleWindowSize, speckleRange, mode);
    cv::Ptr<cv::StereoMatcher> right_matcher = cv::ximgproc::createRightMatcher(left_matcher);;
    cv::Rect ROI;
    left_matcher->compute(im1, im2, left_disparity);
    right_matcher ->compute(im2, im1, right_disparity);
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter = cv::ximgproc::createDisparityWLSFilterGeneric(left_matcher);
    wls_filter->setLambda(lambda);
    wls_filter->setSigmaColor(sigma);
    wls_filter->filter(left_disparity, im1, filtered_disp, right_disparity);
    conf_map = wls_filter->getConfidenceMap();
    ROI = wls_filter->getROI();
    cv::Mat raw_disp_vis, filtered_disp_vis;

    cv::ximgproc::getDisparityVis(left_disparity,raw_disp_vis,vis_mult);
    cv::imwrite("../working_images/raw_disparity.jpg", raw_disp_vis);
    normalize(filtered_disp_vis, filtered_disp_vis, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::ximgproc::getDisparityVis(filtered_disp,filtered_disp_vis,vis_mult);
    cv::imwrite("../working_images/filtered_disparity.jpg", filtered_disp_vis);
    cv::Mat imgCalorBONE;
    applyColorMap(filtered_disp_vis, imgCalorBONE, cv::COLORMAP_BONE);
    cv::imwrite("../working_images/filtered_disparity_bone.jpg", imgCalorBONE);
}


void save(const cv::Mat& image3D, const std::string& fileName)
{
    cv::Mat im1 = cv::imread("../working_images/undistorted_left.jpg", cv::IMREAD_GRAYSCALE);
    std::ofstream outFile(fileName);
    if (!outFile.is_open())
    {
        std::cerr << "ERROR: Could not open " << fileName << std::endl;
        return;
    }
    for (int i = 0; i < image3D.rows; i++)
    {
        const auto* image3D_ptr = image3D.ptr<cv::Vec3f>(i);
        for (int j = 0; j < image3D.cols; j++)
        {
            outFile << image3D_ptr[j][0] << " " << image3D_ptr[j][1] << " " << image3D_ptr[j][2] << " " <<
            static_cast<unsigned>(im1.at<uchar>(i,j)) << " " << static_cast<unsigned>(im1.at<uchar>(i,j)) << " "
            << static_cast<unsigned>(im1.at<uchar>(i,j)) << std::endl;
        }
    }
    outFile.close();
}


void findRTQ(cv::Mat &Q, cv::Mat &camera_matrix, cv::Mat &distortion) {
    int minHessian = 600;
    cv::Mat im1 = cv::imread("../working_images/undistorted_left.jpg");
    cv::Mat im2 = cv::imread("../working_images/undistorted_right.jpg");
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create( minHessian);
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat desc1, desc2;
    detector->detectAndCompute( im1, cv::noArray(), keypoints1, desc1 );
    detector->detectAndCompute( im2, cv::noArray(), keypoints2, desc2 );
    auto* matcher = new cv::BFMatcher(cv::NORM_L2, false);
    std::vector< std::vector<cv::DMatch> > matches_2nn_12, matches_2nn_21;
    matcher->knnMatch( desc1, desc2, matches_2nn_12, 2 );
    matcher->knnMatch( desc2, desc1, matches_2nn_21, 2 );
    std::vector<cv::Point2f> selected_points1, selected_points2;
    const double ratio = 0.8;
    for(auto & i : matches_2nn_12) { // i is queryIdx
        if (i[0].distance / i[1].distance < ratio
            and
            matches_2nn_21[i[0].trainIdx][0].distance
            / matches_2nn_21[i[0].trainIdx][1].distance < ratio) {
            if (matches_2nn_21[i[0].trainIdx][0].trainIdx
                == i[0].queryIdx) {
                selected_points1.push_back(keypoints1[i[0].queryIdx].pt);
                selected_points2.push_back(
                        keypoints2[matches_2nn_21[i[0].trainIdx][0].queryIdx].pt
                );
            }
        }
    }
    cv::Mat Kd;
    camera_matrix.convertTo(Kd, CV_64F);
    cv::Mat mask;
    cv::Mat E = cv::findEssentialMat(selected_points1, selected_points2, Kd.at<double>(0,0),
                                     cv::Point2d(im1.cols/2., im1.rows/2.),
                                     cv::RANSAC, 0.999, 1.0, mask);

    std::vector<cv::Point2f> inlier_match_points1, inlier_match_points2;
    for(int i = 0; i < mask.rows; i++) {
        if(mask.at<unsigned char>(i)){
            inlier_match_points1.push_back(selected_points1[i]);
            inlier_match_points2.push_back(selected_points2[i]);
        }
    }
    mask.release();

    cv::Mat R, t;
    cv::recoverPose(E, inlier_match_points1, inlier_match_points2, R, t, Kd.at<double>(0,0),
                    cv::Point2d(im1.cols/2., im1.rows/2.), mask);
    cv::Mat R1, R2, P1, P2;
    cv::stereoRectify(camera_matrix, distortion, camera_matrix, distortion, im1.size(), R, t, R1, R2, P1, P2, Q,
            cv::CALIB_ZERO_DISPARITY, 1, im1.size());
}


void point_cloud(cv::Mat &Q) {
    cv::Mat im1 = cv::imread("../working_images/undistorted_left.jpg");
    cv::Mat im2 = cv::imread("../working_images/undistorted_right.jpg");
    cv::Mat disp = cv::imread("../working_images/filtered_disparity.jpg", cv::IMREAD_GRAYSCALE);
    std::string path = "../calibration_images";
    cv::Mat image3DOCV, colors;
    reprojectImageTo3D(disp, image3DOCV, Q, true, CV_32F);
    save(image3DOCV, "../points.txt");
}

//void write_to_yml_file(cv::Mat& Matrix, std::string file_path, std::string title) {
//    cv::FileStorage fs(file_path, cv::FileStorage::WRITE);
//    fs << title << Matrix;
//    fs.release();
//}
//
//void read_from_yml_file(cv::Mat& Matrix, std::string file_path, std::string title) {
//    cv::FileStorage fs(file_path, cv::FileStorage::READ);
//    fs[title] >> Matrix;
//    fs.release();
//}


void write_yml_file(cv::Mat& cameraMatrix, cv::Mat& distortionCoefficient, std::string file_path) {
    cv::FileStorage fs(file_path, cv::FileStorage::WRITE);
    fs << "cameraMatrix" << cameraMatrix;
    fs << "distortionCoefficient" << distortionCoefficient;
    fs.release();
}

void read_yml_file(cv::Mat& cameraMatrix, cv::Mat& distortionCoefficient, std::string file_path) {
    cv::FileStorage fs(file_path, cv::FileStorage::READ);
    fs["cameraMatrix"] >> cameraMatrix;
    fs["distortionCoefficient"] >> distortionCoefficient;
    fs.release();
}

int main(int argc, char * argv[]) {
    // conf read
    std::string config_file;
    if (argc == 2){config_file = argv[1];}
    else if (argc == 1){config_file = "../conf.txt";}
    else{throw std::runtime_error("Incorrect amount of arguments!");}

    Configuration configuration;
    std::ifstream config_stream(config_file);
    if(!config_stream.is_open()){
        throw std::runtime_error("Failed to open configurations file " + config_file);
    }
    configuration = read_configuration(config_stream);


    auto start_time = get_current_time_fenced();

    cv::Mat cameraMatrix, distortionCoefficient;

    if (configuration.with_calibration){
        calibration_process(cameraMatrix, distortionCoefficient);
        write_yml_file(cameraMatrix, distortionCoefficient, "../CalibrationMatrices.yml");
    } else {
        read_yml_file(cameraMatrix, distortionCoefficient, "../CalibrationMatrices.yml");
    }
    if (configuration.find_points){
        undistort(cameraMatrix, distortionCoefficient);
        disparity(configuration);
        cv::Mat Q;
        findRTQ(Q, cameraMatrix, distortionCoefficient);
        point_cloud(Q);
    }

    auto finish_time = get_current_time_fenced();
    auto result_time = to_us(finish_time - start_time)/1000000;
    std::cout << "Total time in seconds: " << result_time <<std::endl;
    return 0;
}
