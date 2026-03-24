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

private:
    Ort::Env _env;
    Ort::Session _session;
    Ort::AllocatorWithDefaultOptions _allocator;
    
    std::vector<const char*> _input_node_names;
    std::vector<const char*> _output_node_names;
    std::vector<int64_t> _input_node_dims;

    std::string _input_name_str;
    std::string _output_name_str;
};