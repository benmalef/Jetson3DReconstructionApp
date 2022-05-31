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

Mat DisparityMap::resizeImage(const Mat image, int up_width, int up_height) {
  Mat resized_up;
  cv::resize(image, resized_up, Size(up_width, up_height), INTER_LINEAR);
  return resized_up;
}
void DisparityMap::on_pushButtonStart_clicked() { execDisparityMapImages(); }

void DisparityMap::on_pushButtonExit_clicked() {
  btnExit = true;
  reject();
}

void DisparityMap::createBlockMatcher() {
  if (userChoice == "StereoBM_CPU") {
    stereoBMcpu = StereoBM::create(numOfDisparities, blockSize);
    wls_filter = ximgproc::createDisparityWLSFilter(stereoBMcpu);
    right_matcher = ximgproc::createRightMatcher(stereoBMcpu);
  }
  if (userChoice == "StereoBM_CUDA") {
    stereoBMcuda = cuda::createStereoBM(numOfDisparities, blockSize);
    wls_filter = ximgproc::createDisparityWLSFilter(stereoBMcuda);
    right_matcher = ximgproc::createRightMatcher(stereoBMcuda);
  }
  if (userChoice == "StereoSGBM_CPU") {
    stereoSGBMcpu = StereoSGBM::create(
        minDisparity, numOfDisparities, blockSize, P1, P2, disp12MaxDiff,
        preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange, mode);
    wls_filter = ximgproc::createDisparityWLSFilter(stereoSGBMcpu);
    right_matcher = ximgproc::createRightMatcher(stereoSGBMcpu);
  }
}

void DisparityMap::setBlockMatcherValues() {
  if (userChoice == "StereoBM_CPU") {
    setStereoBMvalues();
  }
  if (userChoice == "StereoBM_CUDA") {
    setStereoBMcudaValues();
  }
  if (userChoice == "StereoSGBM_CPU") {
    setStereoSGBMvalues();
  }
}

void DisparityMap::setStereoBMvalues() {
  if (blockSize % 2 == 1 && blockSize >= 5 && blockSize <= 255)
    stereoBMcpu->setBlockSize(blockSize);

  stereoBMcpu->setDisp12MaxDiff(disp12MaxDiff);
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

  stereoBMcpu->setDisp12MaxDiff(disp12MaxDiff);
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

void DisparityMap::setStereoSGBMvalues() {
  if (blockSize % 2 == 1 && blockSize >= 5 && blockSize <= 255)
    stereoSGBMcpu->setBlockSize(blockSize);

  stereoSGBMcpu->setDisp12MaxDiff(disp12MaxDiff);
  stereoSGBMcpu->setMinDisparity(minDisparity);
  stereoSGBMcpu->setPreFilterCap(preFilterCap);
  stereoSGBMcpu->setP1(P1);
  stereoSGBMcpu->setP2(P2);

  stereoSGBMcpu->setUniquenessRatio(uniquenessRatio);
  stereoSGBMcpu->setSpeckleRange(speckleRange);
  stereoSGBMcpu->setSpeckleWindowSize(speckleWindowSize);
  /*stereoBMcuda->setPreFilterType(0);*/ // sobel Filter

  if (numOfDisparities % 16 == 0)
    stereoSGBMcpu->setNumDisparities(numOfDisparities);
}

void DisparityMap::computeDisparityMapCuda() {

  auto start = std::chrono::system_clock::now();
  cudaLeftFrame.upload(leftFrame);
  cudaRightFrame.upload(rightFrame);

  stereoBMcuda->compute(cudaLeftFrame, cudaRightFrame, cudaDisparityMap);

  cuda::drawColorDisp(cudaDisparityMap, cudaDrawColorDisparity,
                      numOfDisparities);
  cudaDisparityMap.download(disparityMap);
  cudaDrawColorDisparity.download(disparityMapColor);
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "finished computation at " << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";

  right_matcher->compute(leftFrame, rightFrame, rightDisparityMap);
}

void DisparityMap::computeDisparityMap() {

  if (userChoice == "StereoBM_CPU") {

    auto start = std::chrono::system_clock::now();
    stereoBMcpu->compute(leftFrame, rightFrame, disparityMap);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "finished computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s\n";

    right_matcher->compute(leftFrame, rightFrame, rightDisparityMap);
  }
  if (userChoice == "StereoBM_CUDA") {
    computeDisparityMapCuda();
  }
  if (userChoice == "StereoSGBM_CPU") {
    auto startSGBM = std::chrono::system_clock::now();
    stereoSGBMcpu->compute(leftFrame, rightFrame, disparityMap);
    auto endSGBM = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = endSGBM - startSGBM;
    std::time_t end_time = std::chrono::system_clock::to_time_t(endSGBM);
    std::cout << "finished computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s\n";
    right_matcher->compute(leftFrame, rightFrame, rightDisparityMap);
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
  //  Mat disparityMapVis;
  Mat disparityMapColorMap;
  imshow("LeftFrame", leftFrame);
  imshow("RightFrame", rightFrame);
  imshow("DisparityMap", disparityMapColor);
  //  disparityMapColorMap = disparityMapColor;
  //  disparityMapColorMap.convertTo(disparityMapColorMap,);
  //  applyColorMap(disparityMapColor, disparityMapColorMap, COLORMAP_JET);
  imshow("DisparityMapColormap", disparityMapColor);
  //  ximgproc::getDisparityVis(filteredDisparityMap,
  //  disparityMapVisFiltered, 1.0);
  //  imshow("Filtered DisparityMap", filteredDisparityMap);
}

void DisparityMap::showDisparityMapCPU() {

  Mat disparityMapVis;
  Mat disparityMapVisFiltered;
  //  ximgproc::getDisparityVis(filteredDisparityMap,
  //  disparityMapVisFiltered, 3.0);
  ximgproc::getDisparityVis(disparityMap, disparityMapVis, 1.0);
  imshow("LeftFrame", leftFrame);
  imshow("RightFrame", rightFrame);
  //  imshow("Filtered DisparityMap", disparityMapVisFiltered);
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
    //    filteringDisparityMat();
    showDisparityMap();

    // take screenshot
    // reprojectTo3D();

    char key = waitKey(1);
    if (key == 'q' || this->btnExit == true) {
      break;
    }
  }
}

void DisparityMap::execDisparityMapImages() {

  //  remap(leftFrame, leftFrame, stereoRectification.leftStereoMapX,
  //        stereoRectification.leftStereoMapY, cv::INTER_LANCZOS4,
  //        cv::BORDER_CONSTANT, 0);
  //  remap(rightFrame, rightFrame, stereoRectification.rightStereoMapX,
  //        stereoRectification.rightStereoMapY, cv::INTER_LANCZOS4,
  //        cv::BORDER_CONSTANT, 0);

  cvtColor(rightFrame, rightFrame, COLOR_BGR2GRAY);
  cvtColor(leftFrame, leftFrame, COLOR_BGR2GRAY);
  while (true) {
    setBlockMatcherValues();
    computeDisparityMap();
    //    filteringDisparityMat();
    showDisparityMap();

    char key = waitKey(1);
    if (key == 'q' || this->btnExit == true) {
      destroyAllWindows();
      break;
    }
  }
}

void DisparityMap::filteringDisparityMat() {

  wls_filter->setLambda(8000);
  wls_filter->setSigmaColor(3.5);

  wls_filter->filter(disparityMap, leftFrame, filteredDisparityMap,
                     rightDisparityMap);
}

void DisparityMap::on_pushButton_clicked() {
  userOptions();
  createBlockMatcher();
}

void DisparityMap::on_horizontalSliderNumOfDisparities_valueChanged(int value) {
  if (value % 16 == 0) {
    numOfDisparities = value;
    ui->labelNumOfDisparitiesValue->setText(QString::number(value));
  }
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

void DisparityMap::on_horizontalSliderP1_valueChanged(int value) {
  P1 = value;
  ui->labelP1Value->setText(QString::number(value));
}

void DisparityMap::on_horizontalSliderP2_valueChanged(int value) {
  P2 = value;
  ui->labelP2Value->setText(QString::number(value));
}

void DisparityMap::on_pushButtonImportRightImage_clicked() {
  String rightImagePath =
      QFileDialog::getOpenFileName(this, "import right image", "/home")
          .toStdString();
  rightFrame = imread(rightImagePath, IMREAD_COLOR);
  //  rightFrame = resizeImage(rightFrame, 1920, 1080);
  imshow("RightFrame", rightFrame);
}

void DisparityMap::on_pushButtonImportLeftImage_clicked() {
  String leftImagePath =
      QFileDialog::getOpenFileName(this, "import left image", "/home")
          .toStdString();
  leftFrame = imread(leftImagePath, IMREAD_COLOR);
  //  leftFrame = resizeImage(leftFrame, 1920, 1080);
  imshow("LeftFrame", leftFrame);
}

void DisparityMap::on_pushButton_4_clicked() {
  btnExit = false;
  execDisparityMap();
}

void DisparityMap::on_pushButton3DImage_clicked() {
  Mat pointsXYZ;

  cv::reprojectImageTo3D(disparityMap, pointsXYZ, stereoRectification.Q, false,
                         CV_32F);
  viz::writeCloud("/home/benmalef/Desktop/StereoApp/pointCloudStereoBMBen.ply",
                  pointsXYZ, leftFrame);
}

void DisparityMap::on_pushButton_2_clicked() { btnExit = true; }
