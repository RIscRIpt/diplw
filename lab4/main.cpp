#include <iostream>
#include <filesystem>

#include <conio.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std::literals::string_literals;

template<typename F, typename ...Ps>
cv::Mat S(std::string const &stepName, F fn, Ps&&... ps) {
    cv::Mat result = fn(ps...);
    cv::imwrite(stepName + ".png", result);
    return result;
}

cv::Mat makeStructuringElement(int size) {
    return cv::getStructuringElement(cv::MORPH_RECT, cv::Size(size, size));
}

int main() {
    try {
        auto image = cv::imread("0_image.jpg", CV_LOAD_IMAGE_GRAYSCALE);
        if (image.empty())
            throw std::runtime_error("failed to open image");

        cv::threshold(image, image, 250.0, 255.0, CV_THRESH_BINARY);
        cv::imwrite("1_binary.png", image);

        image = 255 - image;
        cv::imwrite("2_inverted.png", image);

        cv::erode(image, image, makeStructuringElement(3));
        cv::imwrite("3_erode.png", image);

        cv::dilate(image, image, makeStructuringElement(7));
        cv::imwrite("4_dilate.png", image);

        cv::erode(image, image, makeStructuringElement(3));
        cv::imwrite("5_erode_and_saveBorder.png", image);

        cv::Mat laplacian;
        cv::Laplacian(image, laplacian, CV_32F);
        cv::normalize(laplacian, laplacian, 0, 255, CV_MINMAX);
        laplacian.convertTo(laplacian, CV_8U);
        cv::threshold(laplacian, laplacian, 127, 255, CV_THRESH_BINARY);
        laplacian = 255 - laplacian;

        cv::imwrite("5_xborder.png", laplacian);

        cv::blur(image, image, { 21, 21 });
        cv::imwrite("6_blur.png", image);

        cv::threshold(image, image, 200.0, 255.0, CV_THRESH_BINARY);
        cv::imwrite("7_binary.png", image);

        cv::dilate(image, image, makeStructuringElement(21));
        cv::imwrite("8_dilate.png", image);

        cv::blur(image, image, { 29, 29 });
        cv::imwrite("9_blur.png", image);

        cv::threshold(image, image, 254.0, 255.0, CV_THRESH_BINARY);
        cv::imwrite("10_binary.png", image);

        cv::dilate(image, image, makeStructuringElement(21));
        cv::imwrite("11_dilate.png", image);

        image += laplacian;
        cv::imwrite("12_addBorder.png", image);

        cv::erode(image, image, makeStructuringElement(3));
        cv::imwrite("13_erode.png", image);

        image -= laplacian;
        cv::imwrite("14_removeBorder.png", image);

        cv::erode(image, image, makeStructuringElement(9));
        cv::imwrite("15_erode.png", image);

        cv::dilate(image, image, makeStructuringElement(9));
        cv::imwrite("16_dilate.png", image);

        cv::medianBlur(image, image, 5);
        cv::imwrite("17_medianBlur.png", image);

        cv::medianBlur(image, image, 5);
        cv::imwrite("18_medianBlur.png", image);

        cv::dilate(image, image, makeStructuringElement(3));
        cv::imwrite("19_dilate.png", image);

        cv::erode(image, image, makeStructuringElement(3));
        cv::imwrite("20_erode.png", image);

        /*
        cv::medianBlur(image, image, 7);
        cv::imwrite("3_blured.png", image);

        cv::dilate(image, image, makeStructuringElement(5));
        cv::imwrite("4_dilate.png", image);

        cv::erode(image, image, makeStructuringElement(11));
        cv::imwrite("5_erode.png", image);

        cv::dilate(image, image, makeStructuringElement(7));
        cv::imwrite("6_dilate.png", image);

        cv::blur(image, image, { 21, 21 });
        cv::imwrite("7_blur.png", image);

        cv::threshold(image, image, 200.0, 255.0, CV_THRESH_BINARY);
        cv::imwrite("8_binary.png", image);

        cv::dilate(image, image, makeStructuringElement(17));
        cv::imwrite("9_dilate.png", image);
        */
    } catch (std::exception const &e) {
        std::cerr << e.what() << '\n';
        getch();
    }
    
    return 0;
}
