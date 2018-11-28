#include <iostream>
#include <conio.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std::literals::string_literals;

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

float distance(cv::Point const &p, cv::Point const &q) {
    cv::Point diff = p - q;
    return cv::sqrt(diff.x * diff.x + diff.y * diff.y);
}

class ffter {
public:
    ffter(cv::Mat const &image)
        : originalSize(image.size())
    {
        cv::Mat paddedImage;
        auto paddedRows = cv::getOptimalDFTSize(image.rows);
        auto paddedCols = cv::getOptimalDFTSize(image.cols);
        cv::copyMakeBorder(image, paddedImage, 0, paddedRows - image.rows, 0, paddedCols - image.cols, cv::BORDER_CONSTANT);

        cv::Mat planes[] = { cv::Mat1f(paddedImage), cv::Mat::zeros(paddedImage.size(), CV_32F) };
        cv::Mat complexImage;
        cv::merge(planes, 2, complexImage);

        cv::dft(complexImage, complexImage);

        cv::split(complexImage, planes);

        real = planes[0];
        imaginary = planes[1];
    }

    cv::Size originalSize;
    cv::Mat real, imaginary;
    cv::Mat magnitude() const {
        cv::Mat magnitude;
        cv::magnitude(real, imaginary, magnitude);
        return magnitude;
    }

    cv::Mat getImage() const {
        cv::Mat planes[] = { real, imaginary };
        cv::Mat complexImage;
        cv::merge(planes, 2, complexImage);

        cv::dft(complexImage, complexImage, cv::DFT_INVERSE);

        cv::split(complexImage, planes);

        cv::normalize(planes[0], planes[0], 0, 1, CV_MINMAX);
        return planes[0](cv::Range(0, originalSize.height), cv::Range(0, originalSize.width));
    }

    ffter& ilpf(float d0) {
        cv::Mat mask = cv::Mat::zeros(originalSize, real.type());
        cv::Point center = originalSize / 2;
        for (int row = center.y - d0; row <= center.y + d0; row++) {
            for (int col = center.x - d0; col <= center.x + d0; col++) {
                cv::Point point{ col, row };
                if (distance(point, center) <= d0)
                    mask.at<float>(point) = 1;
            }
        }
        mask = quadrantFlip(mask);
        real = real.mul(mask);
        imaginary = imaginary.mul(mask);
        return *this;
    }

    ffter& blpf(float d0) {
        cv::Mat mask = cv::Mat::zeros(originalSize, real.type());
        cv::Point center = originalSize / 2;
        for (int row = 0; row < originalSize.height; row++) {
            for (int col = 0; col < originalSize.width; col++) {
                cv::Point point{ col, row };
                float d = distance(point, center);
                mask.at<float>(point) = 1.0f / (1.0f + cv::pow(d / d0, 4));
            }
        }
        mask = quadrantFlip(mask);
        real = real.mul(mask);
        imaginary = imaginary.mul(mask);
        return *this;
    }

    ffter& glpf(float d0) {
        cv::Mat mask = cv::Mat::zeros(originalSize, real.type());
        cv::Point center = originalSize / 2;
        for (int row = 0; row < originalSize.height; row++) {
            for (int col = 0; col < originalSize.width; col++) {
                cv::Point point{ col, row };
                float d = distance(point, center);
                mask.at<float>(point) = cv::exp(-d * d / (2 * d0 * d0));
            }
        }
        mask = quadrantFlip(mask);
        real = real.mul(mask);
        imaginary = imaginary.mul(mask);
        return *this;
    }

    ffter& ihpf(float d0) {
        cv::Mat mask = cv::Mat::ones(originalSize, real.type());
        cv::Point center = originalSize / 2;
        for (int row = center.y - d0; row <= center.y + d0; row++) {
            for (int col = center.x - d0; col <= center.x + d0; col++) {
                cv::Point point{ col, row };
                if (distance(point, center) <= d0)
                    mask.at<float>(point) = 0;
            }
        }
        mask = quadrantFlip(mask);
        real = real.mul(mask);
        imaginary = imaginary.mul(mask);
        return *this;
    }

    ffter& bhpf(float d0) {
        cv::Mat mask = cv::Mat::zeros(originalSize, real.type());
        cv::Point center = originalSize / 2;
        for (int row = 0; row < originalSize.height; row++) {
            for (int col = 0; col < originalSize.width; col++) {
                cv::Point point{ col, row };
                float d = distance(point, center);
                mask.at<float>(point) = 1.0f / (1.0f + cv::pow(d0 / d, 4));
            }
        }
        mask = quadrantFlip(mask);
        real = real.mul(mask);
        imaginary = imaginary.mul(mask);
        return *this;
    }

    ffter& ghpf(float d0) {
        cv::Mat mask = cv::Mat::zeros(originalSize, real.type());
        cv::Point center = originalSize / 2;
        for (int row = 0; row < originalSize.height; row++) {
            for (int col = 0; col < originalSize.width; col++) {
                cv::Point point{ col, row };
                float d = distance(point, center);
                mask.at<float>(point) = 1.0f - cv::exp(-d * d / (2 * d0 * d0));
            }
        }
        mask = quadrantFlip(mask);
        real = real.mul(mask);
        imaginary = imaginary.mul(mask);
        return *this;
    }
};

int main() {
    try {
        auto image = cv::imread("image.png", CV_LOAD_IMAGE_GRAYSCALE);
        if (image.empty())
            throw std::runtime_error("failed to open image");

        auto fft = ffter(image);

        cv::Mat normalizedMagnitude;
        cv::normalize(fft.magnitude(), normalizedMagnitude, 0, 1, CV_MINMAX);

        output_image("fft2.png", normalizedMagnitude);
        normalizedMagnitude = quadrantFlip(normalizedMagnitude);
        output_image("fft2_shifted.png", normalizedMagnitude);

        auto logScaledMagnitude = logScale(fft.magnitude());
        cv::normalize(logScaledMagnitude, logScaledMagnitude, 0, 1, CV_MINMAX);
        logScaledMagnitude = quadrantFlip(logScaledMagnitude);
        output_image("fft2_shifted_logscaled.png", logScaledMagnitude);

        output_image("ifft2.png", fft.getImage());

        std::vector<float> d0s = { 5, 10, 50, 250 };
        for (auto d0 : d0s) {
            auto img = ffter(image);
            output_image("ilpf_"s + std::to_string(d0) + ".png"s, img.ilpf(d0).getImage());
        }

        for (auto d0 : d0s) {
            auto img = ffter(image);
            output_image("blpf_"s + std::to_string(d0) + ".png"s, img.blpf(d0).getImage());
        }

        for (auto d0 : d0s) {
            auto img = ffter(image);
            output_image("glpf_"s + std::to_string(d0) + ".png"s, img.glpf(d0).getImage());
        }

        for (auto d0 : d0s) {
            auto img = ffter(image);
            output_image("ihpf_"s + std::to_string(d0) + ".png"s, img.ihpf(d0).getImage());
        }

        for (auto d0 : d0s) {
            auto img = ffter(image);
            output_image("bhpf_"s + std::to_string(d0) + ".png"s, img.bhpf(d0).getImage());
        }

        for (auto d0 : d0s) {
            auto img = ffter(image);
            output_image("ghpf_"s + std::to_string(d0) + ".png"s, img.ghpf(d0).getImage());
        }

    } catch (std::exception const &e) {
        std::cerr << e.what() << '\n';
    }
    
    return 0;
}
