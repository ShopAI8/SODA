// File: MethodSelector.h

#pragma once 

#include <string>
#include <vector>
#include <memory> 

#include "../third_party/onnxruntime-linux-x64-1.16.3/include/onnxruntime_cxx_api.h"

class MethodSelector {
public:
    MethodSelector(const std::string& model_path);

    // Return float instead of bool to support multiclass and regression models
    float predict(const std::vector<float>& features);

    // Batch prediction API for routing_mode=5
    // Input: batch_features with shape [batch_size, feature_dim]
    // Output: batch prediction results with length batch_size
    std::vector<float> predict_batch(const std::vector<std::vector<float>>& batch_features);

private:
    // Use a static accessor for the global singleton instead of storing Ort::Env _env
    Ort::Session _session{nullptr};
    Ort::AllocatorWithDefaultOptions _allocator;
    
    std::vector<const char*> _input_node_names;
    std::vector<const char*> _output_node_names;
    std::vector<int64_t> _input_node_dims;

    std::string _input_name_str;
    std::string _output_name_str;

    // Retrieve the globally shared ONNX Runtime environment
    static Ort::Env& get_shared_env();
};
