#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

cv::Mat quadrantFlip(cv::Mat const &image) {
    cv::Mat flipped(image.size(), image.type());

    auto rowsCenter = image.rows / 2;
    auto colsCenter = image.cols / 2;

    // Top-left => Bottom-right
    image(cv::Range(0, rowsCenter), cv::Range(0, colsCenter))
        .copyTo(flipped(cv::Range(rowsCenter, image.rows), cv::Range(colsCenter, image.cols)));

    // Top-right => Bottom-left
    image(cv::Range(0, rowsCenter), cv::Range(colsCenter, image.cols))
        .copyTo(flipped(cv::Range(rowsCenter, image.rows), cv::Range(0, colsCenter)));

    // Bottom-left => Top-right
    image(cv::Range(rowsCenter, image.rows), cv::Range(0, colsCenter))
        .copyTo(flipped(cv::Range(0, rowsCenter), cv::Range(colsCenter, image.cols)));

    // Bottom-right => Top-left
    image(cv::Range(rowsCenter, image.rows), cv::Range(colsCenter, image.cols))
        .copyTo(flipped(cv::Range(0, rowsCenter), cv::Range(0, colsCenter)));

    return flipped;
}

// Re(FFT), Im(FFT), Magnitude(FFT)
std::tuple<cv::Mat, cv::Mat, cv::Mat> fft(cv::Mat const &image) {
    cv::Mat paddedImage;
    auto paddedRows = cv::getOptimalDFTSize(image.rows);
    auto paddedCols = cv::getOptimalDFTSize(image.cols);
    cv::copyMakeBorder(image, paddedImage, 0, paddedRows - image.rows, 0, paddedCols - image.cols, cv::BORDER_CONSTANT);

    cv::Mat planes[] = { cv::Mat1f(paddedImage), cv::Mat::zeros(paddedImage.size(), CV_32F) };
    cv::Mat complexImage;
    cv::merge(planes, 2, complexImage);

    cv::dft(complexImage, complexImage);

    cv::split(complexImage, planes);
    cv::Mat magnitudeImage;
    cv::magnitude(planes[0], planes[1], magnitudeImage);

    return { planes[0], planes[1], magnitudeImage };
}

// Re(FFT), Im(FFT), Magnitude(FFT)
cv::Mat ifft(std::tuple<cv::Mat, cv::Mat, cv::Mat> const &fftImage, cv::Size size) {
    cv::Mat planes[] = { std::get<0>(fftImage), std::get<1>(fftImage) };
    cv::Mat complexImage;
    cv::merge(planes, 2, complexImage);

    cv::dft(complexImage, complexImage, cv::DFT_INVERSE);

    cv::split(complexImage, planes);

    cv::normalize(planes[0], planes[0], 0, 1, CV_MINMAX);
    return planes[0](cv::Range(0, size.height), cv::Range(0, size.width));
}

auto logScale(cv::Mat const &image) {
    cv::Mat scaled;
    image.copyTo(scaled);
    scaled += cv::Scalar::all(1);
    cv::log(scaled, scaled);
    return scaled;
}

auto iLogScale(cv::Mat const &image) {
    cv::Mat scaled;
    image.copyTo(scaled);
    cv::exp(scaled, scaled);
    scaled -= cv::Scalar::all(1);
    return scaled;
}

void output_image(std::string const &filename, cv::Mat const &image) {
    cv::Mat output;
    cv::normalize(image, output, 0, 255, CV_MINMAX);
    output.convertTo(output, CV_8U);
    cv::imwrite(filename, output);
}

int main() {
    try {
        auto image = cv::imread("image.png", CV_LOAD_IMAGE_GRAYSCALE);
        if (image.empty())
            throw std::runtime_error("failed to open image");

        //std::tuple<cv::Mat, cv::Mat, cv::Mat> fft(cv::Mat const &image);
        auto fftImage = fft(image);

        cv::Mat magnitude = std::get<2>(fftImage);

        cv::Mat normalizedMagnitude;
        cv::normalize(magnitude, normalizedMagnitude, 0, 1, CV_MINMAX);

        auto logScaledMagnitude = logScale(magnitude);
        cv::normalize(logScaledMagnitude, logScaledMagnitude, 0, 1, CV_MINMAX);

        normalizedMagnitude = quadrantFlip(normalizedMagnitude);
        logScaledMagnitude = quadrantFlip(logScaledMagnitude);

        cv::imshow("normalizedMagnitude", normalizedMagnitude);
        cv::imshow("logScaledMagnitude", logScaledMagnitude);

        output_image("justnormalized.png", normalizedMagnitude);
        output_image("logscaled.png", logScaledMagnitude);

        cv::waitKey();
    } catch (std::exception const &e) {
        std::cerr << e.what() << '\n';

        getchar();
    }

    return 0;
}
