#include "include/MethodSelector.h"
#include <iostream>
#include <stdexcept>

MethodSelector::MethodSelector(const std::string &model_path)
    : _env(ORT_LOGGING_LEVEL_WARNING, "MethodSelector"),
      _session(nullptr)
{
   Ort::SessionOptions session_options;
   session_options.SetIntraOpNumThreads(1);
   session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

   std::cout << "  - [Selector] Initializing ONNX Runtime session..." << std::endl;
   try
   {
      _session = Ort::Session(_env, model_path.c_str(), session_options);
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

// 【修改点】返回值改为 float
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
   } else if (type_info == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
       float* pred_ptr = output_tensor.GetTensorMutableData<float>();
       final_prediction = pred_ptr[0];
   } else {
       throw std::runtime_error("Unsupported ONNX output type!");
   }

   return final_prediction;
}