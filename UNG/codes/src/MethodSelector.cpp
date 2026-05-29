#include "include/MethodSelector.h"
#include <iostream>
#include <stdexcept>
#include <mutex>

// Use a function-local static variable to provide a thread-safe singleton Env.
// The ONNX environment is initialized once per process and shared by all MethodSelector models.
Ort::Env& MethodSelector::get_shared_env() {
    // Set the log level to WARNING to avoid excessive runtime output
    static Ort::Env shared_env(ORT_LOGGING_LEVEL_WARNING, "GlobalONNXEnv");
    return shared_env;
}

MethodSelector::MethodSelector(const std::string &model_path)
    : _session(nullptr)
{
   Ort::SessionOptions session_options;
   
   // Strictly limit concurrency to one thread across all execution dimensions
   session_options.SetIntraOpNumThreads(1); // Use one thread within each operator, such as matrix multiplication
   session_options.SetInterOpNumThreads(1); // Use one thread across independent execution branches
   session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

   std::cout << "  - [Selector] Initializing ONNX Runtime session for: " << model_path << std::endl;
   try
   {
      // Pass the globally shared Env instance
      _session = Ort::Session(get_shared_env(), model_path.c_str(), session_options);
   }
   catch (const Ort::Exception &e)
   {
      throw std::runtime_error("Failed to load ONNX model: " + std::string(e.what()));
   }
   std::cout << "  - [Selector] Session initialized successfully." << std::endl;

   Ort::AllocatedStringPtr input_name_ptr = _session.GetInputNameAllocated(0, _allocator);
   _input_name_str = input_name_ptr.get();
   _input_node_names.push_back(_input_name_str.c_str());

   Ort::AllocatedStringPtr output_name_ptr = _session.GetOutputNameAllocated(0, _allocator);
   _output_name_str = output_name_ptr.get();
   _output_node_names.push_back(_output_name_str.c_str());

   Ort::TypeInfo type_info = _session.GetInputTypeInfo(0);
   auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
   _input_node_dims = tensor_info.GetShape();

   if (_input_node_dims[0] < 0)
   {
      _input_node_dims[0] = 1;
   }
}

float MethodSelector::predict(const std::vector<float> &features)
{
   if (features.size() != _input_node_dims[1])
   {
      throw std::runtime_error("Feature size mismatch. Model expects " +
                               std::to_string(_input_node_dims[1]) +
                               " features, but got " + std::to_string(features.size()));
   }

   Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

   Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                             const_cast<float *>(features.data()),
                                                             features.size(),
                                                             _input_node_dims.data(),
                                                             _input_node_dims.size());

   auto output_tensors = _session.Run(Ort::RunOptions{nullptr},
                                      _input_node_names.data(),
                                      &input_tensor,
                                      1, 
                                      _output_node_names.data(),
                                      1); 

   // Dynamically inspect the ONNX output type to support both classifiers (INT64) and regressors (FLOAT)
   Ort::Value& output_tensor = output_tensors.front();
   auto type_info = output_tensor.GetTensorTypeAndShapeInfo().GetElementType();

   float final_prediction = 0.0f;

   if (type_info == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
       int64_t* pred_ptr = output_tensor.GetTensorMutableData<int64_t>();
       final_prediction = static_cast<float>(pred_ptr[0]);
   } else if (type_info == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
       float* pred_ptr = output_tensor.GetTensorMutableData<float>();
       final_prediction = pred_ptr[0];
   } else {
       throw std::runtime_error("Unsupported ONNX output type!");
   }

   return final_prediction;
}


std::vector<float> MethodSelector::predict_batch(const std::vector<std::vector<float>>& batch_features) {
    if (batch_features.empty()) {
        return {};
    }

    size_t batch_size = batch_features.size();
    size_t feature_dim = batch_features[0].size();

    // 1. Flatten the 2D array into contiguous 1D memory and reserve capacity up front
    std::vector<float> flat_input;
    flat_input.reserve(batch_size * feature_dim);
    for (const auto& row : batch_features) {
        flat_input.insert(flat_input.end(), row.begin(), row.end());
    }

    // 2. Configure the input tensor shape: [batch_size, feature_dim]
    std::vector<int64_t> input_shape = { static_cast<int64_t>(batch_size), static_cast<int64_t>(feature_dim) };

    // 3. Create the ONNX Runtime input tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, 
        flat_input.data(), 
        flat_input.size(), 
        input_shape.data(), 
        input_shape.size()
    );

   // 4. Run inference
    auto output_tensors = _session.Run(
        Ort::RunOptions{nullptr},
        _input_node_names.data(),
        &input_tensor,
        1,
        _output_node_names.data(),
        1
    );

    // 5. Extract batched outputs and perform safe type conversion
    Ort::Value& output_tensor = output_tensors.front();
    auto type_info = output_tensor.GetTensorTypeAndShapeInfo().GetElementType();

    std::vector<float> results(batch_size);

    if (type_info == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        int64_t* output_data = output_tensor.GetTensorMutableData<int64_t>();
        // Safely convert each value to float
        for (size_t i = 0; i < batch_size; ++i) {
            results[i] = static_cast<float>(output_data[i]);
        }
    } else if (type_info == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        float* output_data = output_tensor.GetTensorMutableData<float>();
        // If the output is already float, copy it directly
        results.assign(output_data, output_data + batch_size);
    } else {
        throw std::runtime_error("Unsupported ONNX output type in batch prediction!");
    }

    return results;
}
