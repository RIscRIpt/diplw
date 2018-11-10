#include <iostream>
#include <map>

#include <conio.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

void draw_histogram(std::string const &filename, cv::Mat const &image) {
    int const bins = 256;
    int const channels[] = { 0 };
    int const histSize[] = { bins };
    int const histHeight = 256;
    float const lranges[] = { 0.0f, 256.0f };
    const float* ranges[] = { lranges };

    cv::Mat hist;
    cv::Mat1b histImage = cv::Mat1b::zeros(256 + 5, bins + 4);
    cv::calcHist(&image, 1, channels, {}, hist, 1, histSize, ranges);

    double maxValue;
    cv::minMaxLoc(hist, nullptr, &maxValue);

    for (int i = 0; i < bins; i++) {
        float value = hist.at<float>(i);
        int height = value / maxValue * histHeight;
        cv::line(histImage, cv::Point(i + 2, histHeight - height + 2), cv::Point(i + 2, histHeight + 2), cv::Scalar::all(255));
        cv::line(histImage, cv::Point(i + 2, histHeight - height + 2), cv::Point(i + 2, 2), cv::Scalar::all(128));
    }

    cv::imwrite(filename, histImage);
}

void output_image_and_histogram(std::string const &filename, cv::Mat const &image) {
    cv::imwrite(filename + ".png", image);
    draw_histogram(filename + "_hist.png", image);
}

void log_transform(cv::Mat const &image, cv::Mat &output) {
    cv::Mat imagep1;
    image.convertTo(imagep1, CV_32FC1);
    imagep1 += 1;
    cv::Mat logImage;
    cv::log(imagep1, logImage);
    cv::normalize(logImage, output, 0, 255, cv::NORM_MINMAX);
}

void gamma_correction(cv::Mat const &image, cv::Mat &output, double gamma) {
    cv::Mat lookUpTable(1, 256, CV_8U);
    uchar *p = lookUpTable.ptr();
    for (int i = 0; i < 256; i++) {
        p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    }
    cv::LUT(image, lookUpTable, output);
}

void piecewise_linear_transform(cv::Mat const &image, cv::Mat &output, std::map<uchar, uchar> const &map) {
    if (map.find(0) == map.end() || map.find(255) == map.end())
        throw std::runtime_error("piecewise_linear_transform: map must contain start [0] and end [255] points!");

    cv::Mat lookUpTable(1, 256, CV_8U);
    uchar *p = lookUpTable.ptr();

    auto i = map.cbegin();
    auto nexti = std::next(i);
    do {
        float H = static_cast<float>(nexti->second) - static_cast<float>(i->second);
        float W = static_cast<float>(nexti->first) - static_cast<float>(i->first);
        float coef = -H / W;
        for (int input = i->first; input <= nexti->first; input++) {
            float h = coef * (static_cast<float>(nexti->first) - input);
            p[input] = nexti->second + h;
        }
        i = nexti;
        nexti = std::next(i);
    } while (nexti != map.cend());

    cv::LUT(image, lookUpTable, output);
}

void roberts(cv::Mat const &image, cv::Mat &result) {
    cv::Mat diff = cv::Mat::zeros({ image.rows, image.cols }, CV_32F);
    for (int row = 0; row < image.rows - 1; row++) {
        for (int col = 0; col < image.cols - 1; col++) {
            auto dx = static_cast<float>(image.at<uchar>({ row + 1, col + 1 }))
                - static_cast<float>(image.at<uchar>({ row, col }));
            auto dy = static_cast<float>(image.at<uchar>({ row, col + 1 }))
                - static_cast<float>(image.at<uchar>({ row + 1, col }));
            auto d = sqrt(dx * dx + dy * dy);
            diff.at<float>({ row, col }) = d;
        }
    }
    cv::normalize(diff, diff, 0, 255, cv::NORM_MINMAX);
    diff.convertTo(result, CV_8U);
}

void prewitt(cv::Mat const &image, cv::Mat &result) {
    cv::Mat diff = cv::Mat::zeros({ image.rows, image.cols }, CV_32F);
    for (int row = 1; row < image.rows - 1; row++) {
        for (int col = 1; col < image.cols - 1; col++) {
            auto dx = (static_cast<float>(image.at<uchar>({ row + 1, col - 1 }))
                       + static_cast<float>(image.at<uchar>({ row + 1, col + 0 }))
                       + static_cast<float>(image.at<uchar>({ row + 1, col + 1 }))
                       )
                - (static_cast<float>(image.at<uchar>({ row - 1, col - 1 }))
                   + static_cast<float>(image.at<uchar>({ row - 1, col + 0 }))
                   + static_cast<float>(image.at<uchar>({ row - 1, col + 1 }))
                   );
            auto dy = (static_cast<float>(image.at<uchar>({ row - 1, col + 1 }))
                       + static_cast<float>(image.at<uchar>({ row + 0, col + 1 }))
                       + static_cast<float>(image.at<uchar>({ row + 1, col + 1 }))
                       )
                - (static_cast<float>(image.at<uchar>({ row - 1, col - 1 }))
                   + static_cast<float>(image.at<uchar>({ row + 0, col - 1 }))
                   + static_cast<float>(image.at<uchar>({ row + 1, col - 1 }))
                   );
            auto d = sqrt(dx * dx + dy * dy);
            diff.at<float>({ row, col }) = d;
        }
    }
    cv::normalize(diff, diff, 0, 255, cv::NORM_MINMAX);
    diff.convertTo(result, CV_8U);
}

void sobel(cv::Mat const &image, cv::Mat &result) {
    cv::Mat diff = cv::Mat::zeros({ image.rows, image.cols }, CV_32F);
    for (int row = 1; row < image.rows - 1; row++) {
        for (int col = 1; col < image.cols - 1; col++) {
            auto dx = (static_cast<float>(image.at<uchar>({ row + 1, col - 1 }))
                       + 2 * static_cast<float>(image.at<uchar>({ row + 1, col + 0 }))
                       + static_cast<float>(image.at<uchar>({ row + 1, col + 1 }))
                       )
                - (static_cast<float>(image.at<uchar>({ row - 1, col - 1 }))
                   + 2 * static_cast<float>(image.at<uchar>({ row - 1, col + 0 }))
                   + static_cast<float>(image.at<uchar>({ row - 1, col + 1 }))
                   );
            auto dy = (static_cast<float>(image.at<uchar>({ row - 1, col + 1 }))
                       + 2 * static_cast<float>(image.at<uchar>({ row + 0, col + 1 }))
                       + static_cast<float>(image.at<uchar>({ row + 1, col + 1 }))
                       )
                - (static_cast<float>(image.at<uchar>({ row - 1, col - 1 }))
                   + 2 * static_cast<float>(image.at<uchar>({ row + 0, col - 1 }))
                   + static_cast<float>(image.at<uchar>({ row + 1, col - 1 }))
                   );
            auto d = sqrt(dx * dx + dy * dy);
            diff.at<float>({ row, col }) = d;
        }
    }
    cv::normalize(diff, diff, 0, 255, cv::NORM_MINMAX);
    diff.convertTo(result, CV_8U);
}

auto sd(cv::Mat const &image) {
    cv::Scalar mean, sd;
    cv::meanStdDev(image, mean, sd);
    return sd[0];
}

int main() {
    try {
        auto image = cv::imread("image.png", CV_LOAD_IMAGE_GRAYSCALE);

        draw_histogram("hist.png", image);

        cv::Mat logImage;
        log_transform(image, logImage);
        output_image_and_histogram("log", logImage);

        std::vector<std::pair<double, std::string>> gammaMap{
            { 0.1, "gamma_01" },
            { 0.45, "gamma_045" },
            { 5, "gamma_5" },
        };
        for (auto const &e : gammaMap) {
            cv::Mat gammaImage;
            gamma_correction(image, gammaImage, e.first);
            output_image_and_histogram(e.second, gammaImage);
        }

        std::map<uchar, uchar> pwlfMap{
            { 0, 255 },
            { 100, 200 },
            { 150, 25 },
            { 255, 0 },
        };
        cv::Mat transformedImage;
        piecewise_linear_transform(image, transformedImage, pwlfMap);
        output_image_and_histogram("pwlft", transformedImage);

        cv::Mat eqHistImage;
        cv::equalizeHist(image, eqHistImage);
        output_image_and_histogram("eqhist", eqHistImage);

        std::vector<std::pair<int, std::string>> blurMap{
            { 3, "blur_3" },
            { 15, "blur_15" },
            { 35, "blur_35" },
        };
        for (auto const &e : blurMap) {
            cv::Mat blurImage;
            cv::blur(image, blurImage, { e.first, e.first });
            output_image_and_histogram(e.second, blurImage);
        }

        cv::Mat laplacianBorder;
        cv::Laplacian(image, laplacianBorder, CV_16S);
        cv::Mat laplacianImage;
        cv::subtract(image, laplacianBorder, laplacianImage, {}, CV_8U);
        output_image_and_histogram("laplacian", laplacianImage);

        std::vector<std::pair<int, std::string>> medianMap{
            { 3, "median_3" },
            { 9, "median_9" },
            { 15, "median_15" },
        };
        for (auto const &e : medianMap) {
            cv::Mat medianImage;
            cv::medianBlur(image, medianImage, e.first);
            output_image_and_histogram(e.second, medianImage);
        }

        cv::Mat robertsImage;
        roberts(image, robertsImage);
        output_image_and_histogram("roberts", robertsImage);

        cv::Mat prewittImage;
        prewitt(image, prewittImage);
        output_image_and_histogram("prewitt", prewittImage);

        cv::Mat sobelImage;
        sobel(image, sobelImage);
        output_image_and_histogram("sobel", sobelImage);

        cv::Mat sobelImage2;
        cv::Sobel(image, sobelImage2, 0, 1, 1, 3);
        output_image_and_histogram("sobel2", sobelImage2);

        cv::Mat noisedImage;
        cv::Mat noiseImage(image.rows, image.cols, CV_8S);
        cv::randn(noiseImage, 1, 4);
        cv::add(image, noiseImage, noisedImage, {}, CV_8U);
        output_image_and_histogram("noised", noisedImage);

        std::cout << "Clear image:   " << sd(image) << '\n';
        std::cout << "Noised image:  " << sd(noisedImage) << '\n';

        cv::Mat clearedImage;
        cv::medianBlur(noisedImage, clearedImage, 3);
        output_image_and_histogram("noise_cleared", clearedImage);

        std::cout << "Cleared Image: " << sd(clearedImage) << '\n';

    } catch (std::exception const &e) {
        std::cerr << e.what() << '\n';
    }

    getch();

    return 0;
}
