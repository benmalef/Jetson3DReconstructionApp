#ifndef DISPARITYMAP_H
#define DISPARITYMAP_H

#include "disparitymap.h"
#include "stereocamera.h"
#include "stereorectification.h"
#include "ui_disparitymap.h"
#include <QDialog>
#include <QFileDialog>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/viz/viz3d.hpp>
#include <opencv2/viz/vizcore.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>

namespace Ui {
class DisparityMap;
}

class DisparityMap : public QDialog {
  Q_OBJECT

public:
  explicit DisparityMap(QWidget *parent = nullptr);
  ~DisparityMap();

  void userOptions();
  void execDisparityMap();
  void computeDisparityMap();
  void showDisparityMap();
  void createBlockMatcher();
  void showDisparityMapCPU();
  void setBlockMatcherValues();
  void setStereoBMvalues();
  void computeDisparityMapCuda();
  void showDisparityMapCuda();
  void setStereoBMcudaValues();

  void setStereoSGBMvalues();
  void filteringDisparityMat();
  void execDisparityMapImages();
  Mat resizeImage(const Mat image, int up_width, int up_height);
private slots:
  void on_pushButton_2_clicked();

private slots:
  void on_pushButton3DImage_clicked();

private slots:
  void on_pushButton_4_clicked();

private slots:
  void on_pushButtonImportLeftImage_clicked();

private slots:
  void on_pushButtonImportRightImage_clicked();

private slots:
  void on_horizontalSliderP2_valueChanged(int value);
  void on_horizontalSliderP1_valueChanged(int value);
  void on_horizontalSliderPreFilterSize_valueChanged(int value);
  void on_horizontalSliderTextureThreshold_valueChanged(int value);
  void on_horizontalSliderPreFilterCap_valueChanged(int value);
  void on_horizontalSliderMinDisparity_valueChanged(int value);
  void on_horizontalSliderUniquessRatio_valueChanged(int value);
  void on_horizontalSliderDisp12Max_valueChanged(int value);
  void on_horizontalSliderspeckleWindowSize_valueChanged(int value);
  void on_horizontalSliderSpeckleRange_valueChanged(int value);
  void on_horizontalSliderBlocjSize_valueChanged(int value);
  void on_horizontalSliderNumOfDisparities_valueChanged(int value);
  void on_pushButton_clicked();
  void on_pushButtonExit_clicked();
  void on_pushButtonStart_clicked();

private:
  Ui::DisparityMap *ui;
  StereoRectification stereoRectification;

  // CPU
  Ptr<StereoBM> stereoBMcpu;
  Ptr<StereoSGBM> stereoSGBMcpu;

  Mat leftFrame;
  Mat rightFrame;
  Mat disparityMap;
  Mat tmpFrame;
  // CUDA
  Ptr<cuda::StereoBM> stereoBMcuda;
  cuda::GpuMat cudaLeftFrame, cudaRightFrame;
  cuda::GpuMat cudaDisparityMap;
  cuda::GpuMat cudaDrawColorDisparity;
  Mat disparityMapColor;

  int minDisparity = 0; // normally is 0
  int numOfDisparities = 128;
  int blockSize = 5; // 3-11 range, must be odd and >1
  int P1 = 0; // 8*number_of_image_channels*blockSize*blockSize // controlling
              // the smoothness
  int P2 = 0; // 32*number_of_image_channels*blockSize*blockSize
  int disp12MaxDiff = 0;
  //
  int uniquenessRatio = 0;   // 5-15 range
  int speckleWindowSize = 0; // 50-200 range
  int speckleRange = 0;
  int textureThreshold;
  int preFilterCap = 1;
  int preFilterSize = 5;
  int mode = 0;
  String userChoice;
  int btnExit = false;

  // filter
  Ptr<StereoMatcher> right_matcher;
  Ptr<ximgproc::DisparityWLSFilter> wls_filter;
  Mat rightDisparityMap;
  Mat filteredDisparityMap;
};

#endif // DISPARITYMAP_H
