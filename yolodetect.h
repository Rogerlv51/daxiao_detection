#ifndef YOLODETECT_H
#define YOLODETECT_H

// Cpp native
#include <fstream>
#include <vector>
#include <string>
#include <random>

// OpenCV / DNN / Inference
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

struct Detection
{
    int class_id{ 0 };
    std::string className{};
    float confidence{ 0.0 };
    cv::Scalar color{};
    cv::Rect box{};
};

class Inference
{
public:
    void loadModel(const std::string& onnxModelPath, const cv::Size& modelInputShape, const bool& runWithCuda);
    std::vector<Detection> runInference(const cv::Mat& input);

private:
    
    void loadOnnxNetwork();
    cv::Mat formatToSquare(const cv::Mat& source);

    std::string modelPath{};
    bool cudaEnabled{};

    std::vector<std::string> classes{ "datou", "xiaotou"};

    cv::Size2f modelShape{};

    float modelConfidenceThreshold{ 0.25 };
    float modelScoreThreshold{ 0.40 };
    float modelNMSThreshold{ 0.50 };

    bool letterBoxForSquare = true;

    cv::dnn::Net net;
};

#endif // YOLODETECT_H