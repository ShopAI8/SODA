#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <sstream>

#include <boost/program_options.hpp>

#include "rabitq_side_index.h"
#include "utils.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;

int main(int argc, char** argv) {
    std::string side_file;
    std::string meta_file;
    bool include_rotator_state = false;
    bool update_meta = false;

    try {
        po::options_description desc{"Arguments"};
        desc.add_options()
            ("help,h", "Print usage")
            ("side_file", po::value<std::string>(&side_file)->required(), "Path to rabitq_side.bin")
            ("meta_file", po::value<std::string>(&meta_file)->default_value(""),
             "Path to meta file. Empty means <dirname(side_file)>/meta")
            ("include_rotator_state", po::value<bool>(&include_rotator_state)->default_value(false),
             "Whether to include serialized rotator state in memory estimate")
            ("update_meta", po::value<bool>(&update_meta)->default_value(false),
             "Whether to overwrite rabitq_side_size_bytes in meta");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return 0;
        }
        po::notify(vm);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return 1;
    }

    ANNS::rabitq::RabitQSideIndex side_index;
    if (!side_index.load(side_file)) {
        std::cerr << "Failed to load RabitQ side index: " << side_file << std::endl;
        return 1;
    }

    uint64_t file_size_bytes = 0;
    try {
        file_size_bytes = static_cast<uint64_t>(fs::file_size(side_file));
    } catch (const std::exception&) {
        file_size_bytes = 0;
    }

    const uint64_t memory_est_bytes = side_index.estimated_memory_bytes(include_rotator_state);
    const uint64_t rotator_bytes = side_index.estimated_rotator_state_bytes();

    auto to_mb = [](uint64_t bytes) -> double {
        return static_cast<double>(bytes) / (1024.0 * 1024.0);
    };

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "side_file=" << side_file << "\n";
    std::cout << "file_size_bytes=" << file_size_bytes << "\n";
    std::cout << "file_size_mb=" << to_mb(file_size_bytes) << "\n";
    std::cout << "memory_est_bytes=" << memory_est_bytes << "\n";
    std::cout << "memory_est_mb=" << to_mb(memory_est_bytes) << "\n";
    std::cout << "rotator_state_bytes=" << rotator_bytes << "\n";
    std::cout << "include_rotator_state=" << (include_rotator_state ? 1 : 0) << "\n";

    if (update_meta) {
        fs::path meta_path = meta_file.empty() ? (fs::path(side_file).parent_path() / "meta") : fs::path(meta_file);
        auto meta_data = ANNS::parse_kv_file(meta_path.string());
        if (meta_data.empty()) {
            std::cerr << "Failed to parse meta file: " << meta_path << std::endl;
            return 1;
        }
        meta_data["rabitq_side_size_bytes"] = std::to_string(memory_est_bytes);
        auto it_without = meta_data.find("index_size_without_rabitq(MB)");
        if (it_without != meta_data.end()) {
            try {
                const double without_mb = std::stod(it_without->second);
                const double with_mb = without_mb + to_mb(memory_est_bytes);
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(6) << with_mb;
                meta_data["index_size_with_rabitq(MB)"] = oss.str();
            } catch (const std::exception&) {
                // Keep existing value if parsing fails.
            }
        }
        ANNS::write_kv_file(meta_path.string(), meta_data);
        std::cout << "meta_updated=1\n";
        std::cout << "meta_file=" << meta_path.string() << "\n";
    } else {
        std::cout << "meta_updated=0\n";
    }
    return 0;
}
