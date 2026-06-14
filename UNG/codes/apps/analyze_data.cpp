#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <random>
#include <sstream> // Required for stringstream parsing.
#include <unordered_map>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include "utils.h"
#include "include/uni_nav_graph.h" 

namespace po = boost::program_options;
namespace fs = boost::filesystem;

// ================== Local safe parser ==================
/**
 * @brief Safe local parser for `meta` key-value files.
 * This parser only splits on '=' and stores raw string pairs in a map.
 * It does not perform numeric conversion, which avoids premature `stoul` failures.
 * It also handles Windows `\r\n` line endings.
 */
std::map<std::string, std::string> parse_kv_file_safe(const std::string& filename) {
    std::map<std::string, std::string> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("安全解析器：无法打开 meta 文件: " + filename);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // Strip `\r` from Windows line endings.
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        std::stringstream ss(line);
        std::string key, value;
        if (std::getline(ss, key, '=') && std::getline(ss, value)) {
            // Add whitespace trimming here if needed. For now we assume clean input.
            data[key] = value;
        }
    }
    file.close();
    return data;
}
// ================== End parser ==================


// Helper: compute the mean value.
double calculate_mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    return sum / v.size();
}

// Helper: write a histogram map to CSV.
template<typename K, typename V>
void save_histogram_to_csv(const std::string& filename, const std::map<K, V>& histogram, const std::string& key_header, const std::string& value_header) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "错误：无法打开文件 " << filename << std::endl;
        return;
    }
    outfile << key_header << "," << value_header << "\n";
    for (const auto& pair : histogram) {
        outfile << pair.first << "," << pair.second << "\n";
    }
    outfile.close();
    std::cout << "  - 分析结果已保存到: " << filename << std::endl;
}

// Helper: safely map `old_id` to `new_id`.
ANNS::IdxType get_new_id(ANNS::IdxType old_id, const std::unordered_map<ANNS::IdxType, ANNS::IdxType>& old_to_new_map) {
    auto it = old_to_new_map.find(old_id);
    if (it == old_to_new_map.end()) {
        // This is a hard failure: the ID mapping is incomplete.
        throw std::runtime_error("错误：在 old_to_new_map 中未找到 old_id: " + std::to_string(old_id));
    }
    return it->second;
}

int main(int argc, char** argv) {
    std::string index_path_prefix, base_bin_file, data_type, dist_fn, output_prefix;

    try {
        po::options_description desc{"分析工具参数"};
        desc.add_options()
            ("help,h", "打印帮助信息")
            ("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(), "包含索引文件（meta, .dat 等）的目录路径")
            ("base_bin_file", po::value<std::string>(&base_bin_file)->required(), "已重排序的 base vectors 文件路径 (例如 vecs.bin)")
            ("data_type", po::value<std::string>(&data_type)->required(), "数据类型 <int8/uint8/float>")
            ("dist_fn", po::value<std::string>(&dist_fn)->required(), "距离函数 <L2/IP/cosine>")
            ("output_prefix", po::value<std::string>(&output_prefix)->required(), "保存分析结果CSV文件的前缀 (例如 ./celeba_analysis)");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    } catch (const std::exception& ex) {
        std::cerr << "参数错误: " << ex.what() << std::endl;
        return 1;
    }

    std::cout << "--- 数据集特征分析开始 ---" << std::endl;
    std::cout << "加载索引: " << index_path_prefix << std::endl;

    // --- 0. Load all required inputs ---
    std::map<std::string, std::string> meta_data;
    std::vector<std::vector<ANNS::LabelType>> group_id_to_label_set;
    std::vector<std::vector<ANNS::IdxType>> group_id_to_vec_ids; // Stores OLD IDs.
    std::vector<double> lng_coverage_ratio;
    std::vector<ANNS::IdxType> new_to_old_vec_ids;
    std::shared_ptr<ANNS::IStorage> base_storage;
    std::shared_ptr<ANNS::DistanceHandler> distance_handler;
    ANNS::IdxType num_points = 0;
    ANNS::IdxType num_groups = 0;

    try {
        // ================== Use the local safe parser ==================
        meta_data = parse_kv_file_safe(index_path_prefix + "meta");
        // ================== End local parser block ==================

        // `stoul` is safe here because the map now contains plain text only.
        num_points = std::stoul(meta_data.at("num_points")); // `.at()` gives a clearer error if the key is missing.
        num_groups = std::stoul(meta_data.at("num_groups"));
        std::cout << "  - meta: num_points=" << num_points << ", num_groups=" << num_groups << std::endl;

        ANNS::load_2d_vectors(index_path_prefix + "group_id_to_label_set", group_id_to_label_set);
        std::cout << "  - group_id_to_label_set 加载完毕" << std::endl;

        ANNS::load_2d_vectors(index_path_prefix + "group_id_to_vec_ids.dat", group_id_to_vec_ids);
        std::cout << "  - group_id_to_vec_ids 加载完毕" << std::endl;
        
        ANNS::load_1d_vector(index_path_prefix + "lng_coverage_ratio", lng_coverage_ratio);
        std::cout << "  - lng_coverage_ratio 加载完毕" << std::endl;

        ANNS::load_1d_vector(index_path_prefix + "new_to_old_vec_ids", new_to_old_vec_ids);
        std::cout << "  - new_to_old_vec_ids 加载完毕" << std::endl;

        base_storage = ANNS::create_storage(data_type, false);
        base_storage->load_from_file(base_bin_file, ""); // No label file is needed for this analysis.
        std::cout << "  - base_storage (vecs.bin) 加载完毕" << std::endl;

        distance_handler = ANNS::get_distance_handler(data_type, dist_fn);
        std::cout << "  - distance_handler 初始化完毕" << std::endl;

    } catch (const std::exception& ex) {
        std::cerr << "加载数据时出错: " << ex.what() << std::endl;
        std::cerr << "请检查所有文件路径是否正确，以及 'meta' 文件是否包含 'num_points' 和 'num_groups' 键。" << std::endl;
        return 1;
    }

    // --- Analysis 1: label-set length distribution ---
    std::cout << "\n[分析 1: 标签集长度分布]" << std::endl;
    std::map<size_t, size_t> length_histogram;
    for (const auto& label_set : group_id_to_label_set) {
        if (label_set.size() > 0) { // Skip group 0.
            length_histogram[label_set.size()]++;
        }
    }
    save_histogram_to_csv(output_prefix + "_label_set_length_dist.csv", length_histogram, "LabelSetLength", "GroupCount");

    // --- Analysis 2: group popularity (size) distribution ---
    std::cout << "\n[分析 2: 组流行度分布]" << std::endl;
    std::map<size_t, size_t> popularity_histogram;
    size_t total_vecs_in_groups = 0;
    for (const auto& vec_list : group_id_to_vec_ids) {
        if (vec_list.size() > 0) { // Skip group 0.
            popularity_histogram[vec_list.size()]++;
            total_vecs_in_groups += vec_list.size();
        }
    }
    std::cout << "  - 检查: num_points=" << num_points << ", total_vecs_in_groups=" << total_vecs_in_groups << std::endl;
    save_histogram_to_csv(output_prefix + "_group_popularity_dist.csv", popularity_histogram, "GroupSize", "GroupCount");

    // --- Analysis 3: coverage/selectivity distribution ---
    std::cout << "\n[分析 3: 覆盖率分布]" << std::endl;
    std::map<int, size_t> coverage_histogram; // Bucket by percentage.
    for (double ratio : lng_coverage_ratio) {
        if (ratio > 0) {
            int bin = static_cast<int>(std::floor(ratio * 100));
            coverage_histogram[bin]++;
        }
    }
    save_histogram_to_csv(output_prefix + "_coverage_dist.csv", coverage_histogram, "CoveragePercent_Bin", "GroupCount");

    // --- Analysis 4: vector-space clustering (R value) ---
    std::cout << "\n[分析 4: 向量空间聚类性 (R 值)]" << std::endl;
    
    // 4.1 Build the old_id -> new_id map.
    std::unordered_map<ANNS::IdxType, ANNS::IdxType> old_to_new_map;
    old_to_new_map.reserve(new_to_old_vec_ids.size());
    for (ANNS::IdxType new_id = 0; new_id < new_to_old_vec_ids.size(); ++new_id) {
        old_to_new_map[new_to_old_vec_ids[new_id]] = new_id;
    }
    // 4.2 Build the old_id -> group_id map for quick membership checks.
    std::unordered_map<ANNS::IdxType, ANNS::IdxType> old_id_to_group_id;
    old_id_to_group_id.reserve(num_points);
    for (ANNS::IdxType group_id = 1; group_id < group_id_to_vec_ids.size(); ++group_id) {
        if (group_id >= group_id_to_vec_ids.size()) {
             std::cerr << "警告: group_id " << group_id << " 超出了 group_id_to_vec_ids 的范围 " << group_id_to_vec_ids.size() << std::endl;
             continue;
        }
        for (ANNS::IdxType old_id : group_id_to_vec_ids[group_id]) {
            old_id_to_group_id[old_id] = group_id;
        }
    }
    std::cout << "  - 映射构建完毕" << std::endl;

    const int groups_to_sample = 1000;
    const int pairs_per_group = 50;
    const int total_inter_pairs = groups_to_sample * pairs_per_group;
    const auto dim = base_storage->get_dim();
    std::vector<double> intra_distances;
    std::vector<double> inter_distances;
    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<ANNS::IdxType> group_dist(1, num_groups); // Groups start at 1.
    std::uniform_int_distribution<ANNS::IdxType> point_dist(0, num_points - 1);

    // 4.3 Compute intra-group distances.
    std::cout << "  - 计算组内距离 (采样 " << groups_to_sample << " 个组)..." << std::endl;
    intra_distances.reserve(groups_to_sample * pairs_per_group);
    for (int i = 0; i < groups_to_sample; ++i) {
        ANNS::IdxType group_id = group_dist(gen);
        if (group_id >= group_id_to_vec_ids.size()) continue; // Safety check.
        const auto& old_id_list = group_id_to_vec_ids[group_id];
        if (old_id_list.size() < 2) continue;

        std::uniform_int_distribution<size_t> pair_dist(0, old_id_list.size() - 1);
        for (int j = 0; j < pairs_per_group; ++j) {
            ANNS::IdxType old_id_1 = old_id_list[pair_dist(gen)];
            ANNS::IdxType old_id_2 = old_id_list[pair_dist(gen)];
            if (old_id_1 == old_id_2) continue;

            try {
                ANNS::IdxType new_id_1 = get_new_id(old_id_1, old_to_new_map);
                ANNS::IdxType new_id_2 = get_new_id(old_id_2, old_to_new_map);
                const char* v1 = base_storage->get_vector(new_id_1);
                const char* v2 = base_storage->get_vector(new_id_2);
                intra_distances.push_back(distance_handler->compute(v1, v2, dim));
            } catch (const std::exception& ex) {
                std::cerr << "组内距离计算错误: " << ex.what() << std::endl;
            }
        }
    }

    // 4.4 Compute inter-group distances.
    std::cout << "  - 计算组间距离 (采样 " << total_inter_pairs << " 对)..." << std::endl;
    inter_distances.reserve(total_inter_pairs);
    while (inter_distances.size() < total_inter_pairs) {
        ANNS::IdxType new_id_1 = point_dist(gen);
        ANNS::IdxType new_id_2 = point_dist(gen);
        if (new_id_1 == new_id_2) continue;
        if (new_id_1 >= new_to_old_vec_ids.size() || new_id_2 >= new_to_old_vec_ids.size()) continue; // Safety check.

        ANNS::IdxType old_id_1 = new_to_old_vec_ids[new_id_1];
        ANNS::IdxType old_id_2 = new_to_old_vec_ids[new_id_2];

        // Skip pairs that belong to the same group.
        auto it1 = old_id_to_group_id.find(old_id_1);
        auto it2 = old_id_to_group_id.find(old_id_2);
        if (it1 != old_id_to_group_id.end() && it2 != old_id_to_group_id.end() && it1->second == it2->second) {
            continue; // Same group, so skip.
        }

        const char* v1 = base_storage->get_vector(new_id_1);
        const char* v2 = base_storage->get_vector(new_id_2);
        inter_distances.push_back(distance_handler->compute(v1, v2, dim));
    }

    // 4.5 Compute and report the R value.
    double avg_intra_dist = calculate_mean(intra_distances);
    double avg_inter_dist = calculate_mean(inter_distances);
    double r_value = (avg_inter_dist == 0) ? 0 : (avg_intra_dist / avg_inter_dist);

    std::cout << "\n--- 聚类性分析 (R 值) 结果 ---" << std::endl;
    std::cout << "  - 平均组内距离 (Avg Intra-Group Dist): " << std::fixed << std::setprecision(4) << avg_intra_dist << " (采样 " << intra_distances.size() << " 对)" << std::endl;
    std::cout << "  - 平均组间距离 (Avg Inter-Group Dist): " << std::fixed << std::setprecision(4) << avg_inter_dist << " (采样 " << inter_distances.size() << " 对)" << std::endl;
    std::cout << "  - 聚类指标 R (Intra / Inter): " << std::fixed << std::setprecision(4) << r_value << std::endl;
    std::cout << "  - (R 值越小，说明标签的聚类性越好)" << std::endl;

    std::cout << "\n--- 数据集特征分析完毕 ---" << std::endl;

    return 0;
}
