// 文件名: MethodSelector.h

#pragma once 

#include <string>
#include <vector>
#include <memory> 

#include "../third_party/onnxruntime-linux-x64-1.16.3/include/onnxruntime_cxx_api.h"

class MethodSelector {
public:
    MethodSelector(const std::string& model_path);

    // 返回值从 bool 改为 float，以支持多分类和回归模型
    float predict(const std::vector<float>& features);

    // 批量预测接口（用于 routing_mode=5）
    // 输入: batch_features, shape 为 [batch_size, feature_dim]
    // 输出: 批量预测结果，大小为 batch_size
    std::vector<float> predict_batch(const std::vector<std::vector<float>>& batch_features);

private:
    // 移除 Ort::Env _env; 改为使用静态方法获取全局单例
    Ort::Session _session{nullptr};
    Ort::AllocatorWithDefaultOptions _allocator;
    
    std::vector<const char*> _input_node_names;
    std::vector<const char*> _output_node_names;
    std::vector<int64_t> _input_node_dims;

    std::string _input_name_str;
    std::string _output_name_str;

    // 获取全局共享的 ONNX Runtime 环境
    static Ort::Env& get_shared_env();
};