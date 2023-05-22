#include <filesystem>

#include <torch/cuda.h>
#include <torch/script.h>


int main(int argc, const char* argv[]) {
    if(argc < 2) {
        std::cout << "Specify path to a directory with model checkpoint (./models/simple.pt)" << std::endl;
        return 0;
    }

    std::filesystem::path base_path(argv[1]);
    std::filesystem::path model_path{base_path / "models/simple.pt"};

    // Specify appropriate device
    auto target_device = torch::kCPU;
    if (torch::cuda::is_available()) {
        target_device = torch::kCUDA;
    }

    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(model_path.string(), target_device);

        // Generate random input
        auto input = torch::rand({7, 1, 10}, torch::TensorOptions().device(target_device));
        // Wrap input for model
        std::vector<torch::jit::IValue> input_jit({input});
        // Perform forward pass
        auto output = module.forward(input_jit).toTensor();

        std::cout << "Input tensor: " << input << std::endl;
        std::cout << "Output tensor: " << output << std::endl;
    } catch (const c10::Error &e) {
        std::cerr << "Error loading the model: " << e.msg() << std::endl;
        return -1;
    }
}
