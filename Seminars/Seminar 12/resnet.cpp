#include <filesystem>

#include <torch/cuda.h>
#include <torch/script.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

int main(int argc, const char* argv[]) {
    if(argc < 2) {
        std::cout << "Specify path to a directory with model checkpoint (./models/resnet50.pt) and images folder (./images)" << std::endl;
        return 0;
    }

    std::filesystem::path base_path(argv[1]);
    std::filesystem::path model_path{base_path};
    std::filesystem::path images_path{base_path / "images"};

    // Specify appropriate device
    auto target_device = torch::kCPU;
    if (torch::cuda::is_available()) {
        target_device = torch::kCUDA;
        model_path /= "models/resnet50.pt";
    } else {
        model_path /=  "models/resnet50_cpu.pt";
    }

    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(model_path.string(), target_device);
        // Set model into evaluation mode
        module.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.msg() << std::endl;
        return -1;
    }

    // Use OpenCV to read images
    std::vector<cv::Mat> images;
    for(auto& image_path : std::filesystem::directory_iterator{images_path}) {
        cv::Mat image = imread(image_path.path().string(), cv::IMREAD_COLOR);
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

        images.push_back(image);
    }

    // Convert OpenCV images to torch::Tensor
    auto image_tensors = torch::empty(
        {static_cast<int>(images.size()), 224, 224, 3},
        torch::TensorOptions(torch::kUInt8)
    );
    for(int i = 0; i < images.size(); ++i){
        auto image_tensor = torch::from_blob(
                images[i].data, {images[i].rows, images[i].cols, 3},
                torch::TensorOptions(torch::kUInt8)
            );
        image_tensors.select(0, i).copy_(image_tensor);
    }

    // Normalize data
    auto mean = torch::tensor(
            {0.485, 0.456, 0.406},
            torch::TensorOptions(torch::kFloat32)
            ).reshape({1, 3, 1, 1});
    auto std = torch::tensor(
            {0.229, 0.224, 0.225},
            torch::TensorOptions(torch::kFloat32)
            ).reshape({1, 3, 1, 1});
    auto batch = (
        (image_tensors.permute({0, 3, 1, 2}).to(torch::kFloat32) / 255.0 - mean) / std
    ).to(target_device);

    // Wrap input for model
    std::vector<torch::jit::IValue> batch_jit({batch});

    // Perform forward pass
    auto logits = module.forward(batch_jit).toTensor();
    auto predictions = torch::argmax(logits, 1);
    std::cout << "Tensor with predictions: " << predictions << std::endl;

    // Display results
    plt::figure_size(1000, 700);
    for(int i = 0; i < images.size(); ++i){
        std::stringstream ss;
        ss << "Predicted label: " << predictions[i].item<int>();

        plt::subplot(2, images.size() / 2, i + 1);
        plt::imshow(images[i].data, images[i].rows, images[i].cols, 3);
        plt::title(ss.str());
    }
    plt::tight_layout();
    plt::save((base_path / "predictions.png").string(), 500);
    plt::show();
}
