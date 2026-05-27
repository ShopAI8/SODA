#include "include/MethodSelector.h"
#include <iostream>
#include <stdexcept>
#include <mutex>

// 使用局部静态变量实现线程安全的单例 Env。无论实例化多少个 MethodSelector 模型（L1, L2, SmartRoute 等），整个进程都只会初始化一次 ONNX 环境，共用底层的线程池。
Ort::Env& MethodSelector::get_shared_env() {
    // 设置日志级别为 WARNING，避免输出过多无用信息
    static Ort::Env shared_env(ORT_LOGGING_LEVEL_WARNING, "GlobalONNXEnv");
    return shared_env;
}

MethodSelector::MethodSelector(const std::string &model_path)
    : _session(nullptr)
{
   Ort::SessionOptions session_options;
   
   // 严格限制所有维度的并发线程数为 1
   session_options.SetIntraOpNumThreads(1); // 限制算子内（如矩阵乘法）单线程
   session_options.SetInterOpNumThreads(1); // 限制算子间（独立的计算分支）单线程
   session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

   std::cout << "  - [Selector] Initializing ONNX Runtime session for: " << model_path << std::endl;
   try
   {
      // 传入全局共享的 Env 实例
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

   // 动态判断 ONNX 输出类型，兼容分类器(INT64)和回归器(FLOAT)
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

    // 1. 将 2D 数组展平为 1D 连续内存,预分配内存避免多次扩容
    std::vector<float> flat_input;
    flat_input.reserve(batch_size * feature_dim);
    for (const auto& row : batch_features) {
        flat_input.insert(flat_input.end(), row.begin(), row.end());
    }

    // 2. 设置输入 Tensor 的 Shape (维度: [batch_size, feature_dim])
    std::vector<int64_t> input_shape = { static_cast<int64_t>(batch_size), static_cast<int64_t>(feature_dim) };

    // 3. 创建 ONNX Runtime 的输入 Tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, 
        flat_input.data(), 
        flat_input.size(), 
        input_shape.data(), 
        input_shape.size()
    );

   // 4. 运行推理
    auto output_tensors = _session.Run(
        Ort::RunOptions{nullptr},
        _input_node_names.data(),
        &input_tensor,
        1,
        _output_node_names.data(),
        1
    );

    // 5. 提取批量输出结果并进行安全类型转换
    Ort::Value& output_tensor = output_tensors.front();
    auto type_info = output_tensor.GetTensorTypeAndShapeInfo().GetElementType();

    std::vector<float> results(batch_size);

    if (type_info == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        int64_t* output_data = output_tensor.GetTensorMutableData<int64_t>();
        // 逐个安全转换为 float
        for (size_t i = 0; i < batch_size; ++i) {
            results[i] = static_cast<float>(output_data[i]);
        }
    } else if (type_info == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        float* output_data = output_tensor.GetTensorMutableData<float>();
        // 如果本身就是 float，直接拷贝
        results.assign(output_data, output_data + batch_size);
    } else {
        throw std::runtime_error("Unsupported ONNX output type in batch prediction!");
    }

    return results;
}