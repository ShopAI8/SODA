#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

#include "rabitq_side_index_acorn.h"

namespace fs = std::filesystem;

namespace {

std::map<std::string, std::string> parse_kv_file(const std::string& filename) {
    std::map<std::string, std::string> kv;
    std::ifstream in(filename);
    if (!in.is_open()) {
        return kv;
    }
    std::string line;
    while (std::getline(in, line)) {
        std::size_t pos = line.find(':');
        if (pos == std::string::npos) {
            pos = line.find('=');
        }
        if (pos == std::string::npos) {
            continue;
        }
        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);
        auto trim = [](std::string& s) {
            const char* ws = " \t\r\n";
            s.erase(0, s.find_first_not_of(ws));
            s.erase(s.find_last_not_of(ws) + 1);
        };
        trim(key);
        trim(value);
        if (!key.empty()) {
            kv[key] = value;
        }
    }
    return kv;
}

bool write_kv_file(const std::string& filename, const std::map<std::string, std::string>& kv) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        return false;
    }
    for (const auto& [k, v] : kv) {
        out << k << ":" << v << "\n";
    }
    return out.good();
}

}  // namespace

int main(int argc, char** argv) {
    std::string side_file;
    std::string meta_file;
    bool update_meta = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: " << argv[0]
                << " --side_file <path> [--meta_file <path>] [--update_meta true|false]\n";
            return 0;
        }
        if (arg == "--side_file" && i + 1 < argc) {
            side_file = argv[++i];
            continue;
        }
        if (arg == "--meta_file" && i + 1 < argc) {
            meta_file = argv[++i];
            continue;
        }
        if (arg == "--update_meta" && i + 1 < argc) {
            const std::string v = argv[++i];
            update_meta = (v == "1" || v == "true" || v == "TRUE" || v == "True");
            continue;
        }
        std::cerr << "Unknown/invalid argument: " << arg << "\n";
        return 1;
    }
    if (side_file.empty()) {
        std::cerr << "--side_file is required\n";
        return 1;
    }

    acorn_rabitq::RabitQSideIndex side_index;
    if (!side_index.load(side_file)) {
        std::cerr << "Failed to load ACORN RabitQ side index: " << side_file << std::endl;
        return 1;
    }

    const uint64_t memory_est_bytes = side_index.estimated_memory_bytes(false);
    uint64_t file_size_bytes = 0;
    try {
        file_size_bytes = static_cast<uint64_t>(fs::file_size(side_file));
    } catch (const std::exception&) {
        file_size_bytes = 0;
    }

    auto to_mb = [](uint64_t b) { return static_cast<double>(b) / (1024.0 * 1024.0); };
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "side_file=" << side_file << "\n";
    std::cout << "file_size_bytes=" << file_size_bytes << "\n";
    std::cout << "file_size_mb=" << to_mb(file_size_bytes) << "\n";
    std::cout << "memory_est_bytes=" << memory_est_bytes << "\n";
    std::cout << "memory_est_mb=" << to_mb(memory_est_bytes) << "\n";

    if (!update_meta) {
        std::cout << "meta_updated=0\n";
        return 0;
    }

    fs::path meta_path;
    if (!meta_file.empty()) {
        meta_path = fs::path(meta_file);
    } else {
        fs::path side_path(side_file);
        std::string name = side_path.filename().string();
        const std::string suffix = ".rabitq_side.bin";
        if (name.size() <= suffix.size() || name.rfind(suffix) != name.size() - suffix.size()) {
            std::cerr << "Cannot infer meta file from side_file: " << side_file << std::endl;
            return 1;
        }
        const std::string stem = name.substr(0, name.size() - suffix.size());
        meta_path = side_path.parent_path() / (stem + ".meta");
    }

    auto meta = parse_kv_file(meta_path.string());
    if (meta.empty()) {
        std::cerr << "Failed to parse meta file: " << meta_path << std::endl;
        return 1;
    }

    meta["rabitq_side_size_bytes"] = std::to_string(memory_est_bytes);

    try {
        const int rabitq_enabled = std::stoi(meta.count("rabitq_enabled") ? meta["rabitq_enabled"] : "0");
        const uint64_t side_bytes = rabitq_enabled ? memory_est_bytes : 0ULL;
        if (meta.count("index_only_logical_memory_bytes")) {
            const uint64_t idx_only_logical = static_cast<uint64_t>(std::stoull(meta["index_only_logical_memory_bytes"]));
            meta["index_only_size_bytes"] = std::to_string(idx_only_logical + side_bytes);
        }
        if (meta.count("total_logical_memory_bytes")) {
            const uint64_t total_logical = static_cast<uint64_t>(std::stoull(meta["total_logical_memory_bytes"]));
            meta["total_size_bytes"] = std::to_string(total_logical + side_bytes);
        }
    } catch (const std::exception&) {
        // keep existing values if parsing fails
    }

    if (!write_kv_file(meta_path.string(), meta)) {
        std::cerr << "Failed to write meta file: " << meta_path << std::endl;
        return 1;
    }
    std::cout << "meta_updated=1\n";
    std::cout << "meta_file=" << meta_path.string() << "\n";
    return 0;
}
