#include "yolodetect.h"
using namespace std;

int main() {
	bool runOnGPU = false;

	Inference inf;

	inf.loadModel("lineone0401.onnx", cv::Size(640, 640), runOnGPU);


	std::string pattern2 = "line1/val_xiaotou/*.jpg";
	std::vector<cv::String> filenames2;
	int num = 0;
	cv::glob(pattern2, filenames2, false);
	for (const auto& filename : filenames2) {
		cout << "检测图片名称：" << filename << endl;
		cv::Mat img = cv::imread(filename);

		std::vector<Detection> output = inf.runInference(img);
		int detections = output.size();
		if (detections == 0) {
			cout << "未检测到任何物体" << endl;
		}
		else {
			for (int i = 0; i < detections; ++i)
			{
				Detection detection = output[i];

				cv::Rect box = detection.box;
				cv::Scalar color = detection.color;

				// Detection box
				cv::rectangle(img, box, color, 2);

				// Detection box text
				std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
				cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
				cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

				cv::rectangle(img, textBox, color, cv::FILLED);
				cv::putText(img, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);

				cout << "检测到的类别为：" << detection.className << "，置信度为：" << detection.confidence << endl;
			}
			// Inference ends here...

			// This is only for preview purposes
			float scale = 0.8;
			cv::resize(img, img, cv::Size(img.cols * scale, img.rows * scale));
			cv::imshow("Inference", img);

			cv::waitKey(0);
			num++;
		}
		
	}
	cout << "推理数量为：" << num << endl;
	return 0;
}