#include <iostream>
#include <filesystem>

#include <conio.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std::literals::string_literals;

std::vector<cv::Point> points_p{
    { 0, -1 },
    { -1, 0 },
    { +1, 0 },
    { 0, +1 },
};

std::vector<cv::Point> points_x{
    { -1, -1 },
    { +1, -1 },
    { -1, +1 },
    { +1, +1 },
};

std::vector<cv::Point> points_o{
    { -1, -1 },
    {  0, -1 },
    { +1, -1 },
    { -1,  0 },
    { +1,  0 },
    { -1, +1 },
    {  0, +1 },
    { +1, +1 },
};

std::vector<std::pair<cv::Point, std::vector<cv::Point> const&>> point_map{
    { { 20, 15 }, points_p },
    { { 27, 40 }, points_x },
    { { 100, 100 }, points_o },
};

int main(int argc, char *argv[]) {
    auto color_image = cv::imread("image.png", 1);

    cv::imwrite("image2.png", color_image);
    cv::imwrite("image2.jpg", color_image);

    auto png_size = std::filesystem::file_size("image2.png");
    auto jpg_size = std::filesystem::file_size("image2.jpg");

    std::cout << "PNG Ks : " << (color_image.rows * color_image.cols * 24) / 8.0 / static_cast<double>(png_size) << '\n';
    std::cout << "JPG Ks : " << (color_image.rows * color_image.cols * 24) / 8.0 / static_cast<double>(jpg_size) << '\n';

    cv::Mat gray_image;
    cv::cvtColor(color_image, gray_image, CV_BGR2GRAY);

    cv::imwrite("gray.png", gray_image);

    cv::Mat gray_image_binary;
    cv::threshold(gray_image, gray_image_binary, 255.0 * 0.25, 255.0, CV_THRESH_BINARY);
    cv::imwrite("gray_binary_25.png", gray_image_binary);
    cv::threshold(gray_image, gray_image_binary, 255.0 * 0.50, 255.0, CV_THRESH_BINARY);
    cv::imwrite("gray_binary_50.png", gray_image_binary);
    cv::threshold(gray_image, gray_image_binary, 255.0 * 0.75, 255.0, CV_THRESH_BINARY);
    cv::imwrite("gray_binary_75.png", gray_image_binary);


    cv::Mat gray_plane;
    std::string gray_plane_filename = "gray_plane_X.png";
    for (int i = 0; i < 8; i++) {
        gray_plane = ((gray_image / (1 << i)) & 1) * 255;
        gray_plane_filename[11] = '0' + i;
        cv::imwrite(gray_plane_filename, gray_plane);
    }

    std::vector<int> kernel_sizes = { 5, 10, 20, 50 };
    for (auto kernel_size : kernel_sizes) {
        auto tmp_img = gray_image.clone();
        for (int row = 0; row < tmp_img.rows; row += kernel_size) {
            for (int col = 0; col < tmp_img.cols; col += kernel_size) {
                cv::Rect rect(col, row, kernel_size, kernel_size);
                tmp_img(rect) = cv::mean(tmp_img(rect));
            }
        }
        cv::imwrite("gray_discret_"s + std::to_string(kernel_size) + ".png", tmp_img);
    }

    cv::imwrite("clip.png", gray_image(cv::Rect(gray_image.cols / 2 - 50, gray_image.rows / 2 - 50, 100, 100)));

    for (auto const &[point, map] : point_map) {
        std::cout << "[" << point << "]: ";
        for (size_t i = 0; i < map.size(); i++) {
            std::cout << static_cast<int>(gray_image.at<uint8_t>(point + map[i])) << ' ';
        }
        std::cout << '\n';
    }

    // Average brightness
    cv::Mat image_hsv, image_v;
    cv::cvtColor(color_image, image_hsv, CV_BGR2HSV);
    cv::extractChannel(image_hsv, image_v, 2);
    double avg_brightness = cv::mean(image_v)[0];
    std::cout << "Average brightness: " << avg_brightness << '\n';

    std::vector<cv::Rect> watermark_map{
        { gray_image.cols / 2 - 10, gray_image.rows / 2 - 10, 20, 20 },
        { 0, 0, 20, 20 },
        { gray_image.cols - 20, 0, 20, 20 },
        { 0, gray_image.rows - 20, 20, 20 },
        { gray_image.cols - 20, gray_image.rows - 20, 20, 20 },
    };

    cv::Mat mark_image = gray_image.clone();
    for (auto const &rect : watermark_map) {
        if (avg_brightness < 128.0) {
            mark_image(rect) = 255.0;
        } else {
            mark_image(rect) = 0.0;
        }
    }
    cv::imwrite("marked.png", mark_image);

    getch();

    return 0;
}
