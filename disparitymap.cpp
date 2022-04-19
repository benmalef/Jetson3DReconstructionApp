#include "disparitymap.h"
#include "ui_disparitymap.h"

DisparityMap::DisparityMap(QWidget *parent)
    : QDialog(parent), ui(new Ui::DisparityMap) {
  ui->setupUi(this);
  userOptions();
  createBlockMatcher();
}

DisparityMap::~DisparityMap() { delete ui; }

void DisparityMap::userOptions() {
  if (ui->radioButtonStereoBM->isChecked()) {
    if (ui->radioButtonCPU->isChecked()) {
      userChoice = "StereoBM_CPU";
    } else if (ui->radioButtonCuda->isChecked()) {
      userChoice = "StereoBM_CUDA";
    }
  }
  if (ui->radioButtonStereoSGBM->isChecked()) {
    if (ui->radioButtonCPU->isChecked()) {
      userChoice = "StereoSGBM_CPU";
    } else if (ui->radioButtonCuda->isChecked()) {
      userChoice = "StereoSGBM_CUDA";
    }
  }
}

void DisparityMap::on_pushButtonStart_clicked() { execDisparityMap(); }

void DisparityMap::on_pushButtonExit_clicked() {
  btnExit = true;
  reject();
}

void DisparityMap::createBlockMatcher() {
  if (userChoice == "StereoBM_CPU") {
    stereoBMcpu = StereoBM::create(numOfDisparities, blockSize);
  }
  if (userChoice == "StereoBM_CUDA") {
    stereoBMcuda = cuda::createStereoBM(numOfDisparities, blockSize);
  }
  if (userChoice == "StereoSGBM_CPU") {
    stereoSGBMcpu = StereoSGBM::create(
        minDisparity, numOfDisparities, blockSize, P1, P2, disp12MaxDiff,
        preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange, mode);
  }
}

void DisparityMap::setBlockMatcherValues() {
  if (userChoice == "StereoBM_CPU") {
    setStereoBMvalues();
  }
  if (userChoice == "StereoBM_CUDA") {
    setStereoBMcudaValues();
  }
}

void DisparityMap::setStereoBMvalues() {
  if (blockSize % 2 == 1 && blockSize >= 5 && blockSize <= 255)
    stereoBMcpu->setBlockSize(blockSize);

  //  stereoBMcpu->setDisp12MaxDiff(disp12MaxDiff);
  stereoBMcpu->setMinDisparity(minDisparity);
  stereoBMcpu->setPreFilterCap(preFilterCap);

  if (preFilterSize % 2 == 1)
    stereoBMcpu->setPreFilterSize(preFilterSize);

  stereoBMcpu->setTextureThreshold(textureThreshold);
  stereoBMcpu->setUniquenessRatio(uniquenessRatio);
  stereoBMcpu->setSpeckleRange(speckleRange);
  stereoBMcpu->setSpeckleWindowSize(speckleWindowSize);
  stereoBMcpu->setPreFilterType(1); // sobel Filter

  if (numOfDisparities % 16 == 0)
    stereoBMcpu->setNumDisparities(numOfDisparities);
}

void DisparityMap::setStereoBMcudaValues() {
  if (blockSize % 2 == 1 && blockSize >= 5 && blockSize <= 255)
    stereoBMcuda->setBlockSize(blockSize);

  //  stereoBMcpu->setDisp12MaxDiff(disp12MaxDiff);
  stereoBMcuda->setMinDisparity(minDisparity);
  stereoBMcuda->setPreFilterCap(preFilterCap);

  if (preFilterSize % 2 == 1)
    stereoBMcuda->setPreFilterSize(preFilterSize);

  stereoBMcuda->setTextureThreshold(textureThreshold);
  stereoBMcuda->setUniquenessRatio(uniquenessRatio);
  stereoBMcuda->setSpeckleRange(speckleRange);
  stereoBMcuda->setSpeckleWindowSize(speckleWindowSize);
  /*stereoBMcuda->setPreFilterType(0);*/ // sobel Filter

  if (numOfDisparities % 16 == 0)
    stereoBMcuda->setNumDisparities(numOfDisparities);
}

void DisparityMap::computeDisparityMapCuda() {

  cudaLeftFrame.upload(leftFrame);
  cudaRightFrame.upload(rightFrame);

  stereoBMcuda->compute(cudaLeftFrame, cudaRightFrame, cudaDisparityMap);

  cuda::drawColorDisp(cudaDisparityMap, cudaDrawColorDisparity,
                      numOfDisparities);
  cudaDisparityMap.download(disparityMap);

  cudaDrawColorDisparity.download(disparityMapColor);
}

void DisparityMap::computeDisparityMap() {

  if (userChoice == "StereoBM_CPU") {
    stereoBMcpu->compute(leftFrame, rightFrame, disparityMap);
  }
  if (userChoice == "StereoBM_CUDA") {
    computeDisparityMapCuda();
  }
  if (userChoice == "StereoSGBM_CPU") {
    stereoSGBMcpu->compute(leftFrame, rightFrame, disparityMap);
  }
}

void DisparityMap::showDisparityMap() {
  if (userChoice == "StereoBM_CPU" || userChoice == "StereoSGBM_CPU") {
    showDisparityMapCPU();
  }
  if (userChoice == "StereoBM_CUDA") {
    showDisparityMapCuda();
  }
}

void DisparityMap::showDisparityMapCuda() {

  imshow("LeftFrame", leftFrame);
  imshow("RightFrame", rightFrame);
  imshow("DisparityMap", disparityMapColor);
}
void DisparityMap::showDisparityMapCPU() {

  Mat disparityMapVis;
  ximgproc::getDisparityVis(disparityMap, disparityMapVis, 1.0);
  imshow("LeftFrame", leftFrame);
  imshow("RightFrame", rightFrame);
  imshow("DisparityMap", disparityMapVis);
}
void DisparityMap::execDisparityMap() {

  StereoCamera stereoCamera;
  StereoRectification stereoRectification;

  while (stereoCamera.isOpened()) {

    stereoCamera.getCapLeft().read(leftFrame);
    stereoCamera.getCapRight().read(rightFrame);

    remap(leftFrame, leftFrame, stereoRectification.leftStereoMapX,
          stereoRectification.leftStereoMapY, cv::INTER_LANCZOS4,
          cv::BORDER_CONSTANT, 0);
    remap(rightFrame, rightFrame, stereoRectification.rightStereoMapX,
          stereoRectification.rightStereoMapY, cv::INTER_LANCZOS4,
          cv::BORDER_CONSTANT, 0);

    cvtColor(rightFrame, rightFrame, COLOR_BGR2GRAY);
    cvtColor(leftFrame, leftFrame, COLOR_BGR2GRAY);

    setBlockMatcherValues();

    computeDisparityMap();

    showDisparityMap();

    // take screenshot
    // reprojectTo3D();

    char key = waitKey(1);
    if (key == 'q' || this->btnExit == true) {
      break;
    }
  }
}

void DisparityMap::on_pushButton_clicked() {
  userOptions();
  createBlockMatcher();
}

void DisparityMap::on_horizontalSliderNumOfDisparities_valueChanged(int value) {
  numOfDisparities = value;
  ui->labelNumOfDisparitiesValue->setText(QString::number(value));
}

void DisparityMap::on_horizontalSliderBlocjSize_valueChanged(int value) {
  blockSize = value;
  ui->labelBlockSizeValue->setText(QString::number(value));
}

void DisparityMap::on_horizontalSliderSpeckleRange_valueChanged(int value) {
  speckleRange = value;
  ui->labelSpeckleRangeValue->setText(QString::number(value));
}

void DisparityMap::on_horizontalSliderspeckleWindowSize_valueChanged(
    int value) {
  speckleWindowSize = value;
  ui->labelSpeckleWindowSizeValue->setText(QString::number(value));
}

void DisparityMap::on_horizontalSliderDisp12Max_valueChanged(int value) {
  disp12MaxDiff = value;
  ui->labelDisp12MaxValue->setText(QString::number(value));
}

void DisparityMap::on_horizontalSliderUniquessRatio_valueChanged(int value) {
  uniquenessRatio = value;
  ui->labelUniquenessRatioValue->setText(QString::number(value));
}

void DisparityMap::on_horizontalSliderMinDisparity_valueChanged(int value) {
  minDisparity = value;
  ui->labelMinDisparityValue->setText(QString::number(value));
}

void DisparityMap::on_horizontalSliderPreFilterCap_valueChanged(int value) {
  preFilterCap = value;
  ui->labelPreFilterCapValue->setText(QString::number(value));
}

void DisparityMap::on_horizontalSliderTextureThreshold_valueChanged(int value) {
  textureThreshold = value;
  ui->labelTextureThresholdValue->setText(QString::number(value));
}

void DisparityMap::on_horizontalSliderPreFilterSize_valueChanged(int value) {
  preFilterSize = value;
  ui->labelPreFilterSizeValue->setText(QString::number(value));
}
