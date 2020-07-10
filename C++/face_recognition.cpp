#include "opencv2/opencv.hpp"
#include <iostream>
#include <sstream>


int main() {
    const cv::String protoFile = "deploy.prototxt.txt";
    const cv::String weightsFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel";


    cv::dnn::Net model = cv::dnn::readNetFromCaffe(protoFile, weightsFile);


    cv::Mat image = cv::imread("image.jpg");
    int height = image.size[0];
    int width = image.size[1];
    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0, cv::Size(300, 300), cv::Scalar(104.0, 177.0, 123.0));

    model.setInput(blob);

    cv::Mat output = model.forward();
    cv::Mat outputMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());
    printf("%d, %d", outputMat.size[0], outputMat.size[1]);
    
    for (int i = 0; i < outputMat.rows; i++)
    {
            float confidence = outputMat.at<float>(i, 2);

            if (confidence > 0.5)
            {
                    int x1 = static_cast<int>(outputMat.at<float>(i, 3) * width);
                    int y1 = static_cast<int>(outputMat.at<float>(i, 4) * height);
                    int x2 = static_cast<int>(outputMat.at<float>(i, 5) * width);
                    int y2 = static_cast<int>(outputMat.at<float>(i, 6) * height);
                    cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
                    std::stringstream ss;
                    std::string str;
                    ss << confidence;
                    ss >> str;
                    cv::putText(image, str, cv::Point(x1, y1), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 0, 0), 2);
            }
    }
    cv::namedWindow("OpenCV window");

    // Finally, we display our image and ask the program to wait for a key to be pressed
    imshow("OpenCV window", image);
    cv::waitKey(0);

    return 0;
}
