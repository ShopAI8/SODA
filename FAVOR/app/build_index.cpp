#include "dataset.h"
#include "favor.h"
#include <thread>
#include <cctype>
#include <filesystem>
#include <chrono>

static bool IsIntegerString(const std::string& s) {
    if (s.empty()) {
        return false;
    }
    for (char c : s) {
        if (!std::isdigit(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    return true;
}

template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}

int main(int argc, char* argv[]) {
    int num_threads = 32;

    if (argc != 3 && argc != 4 && argc != 5) {
        std::cerr << "Usage: " << argv[0] << " baseset_path [attribute_path] index_path [num_threads]\n";
        return 1;
    }

    std::string baseset_path(argv[1]);
    bool use_attribute = false;
    std::string attribute_path;
    std::string index_path;
    namespace fs = std::filesystem;

    if (argc == 3) {
        index_path = argv[2];
    } else if (argc == 4) {
        if (IsIntegerString(argv[3])) {
            index_path = argv[2];
            num_threads = std::stoi(argv[3]);
        } else if (fs::exists(argv[2]) && fs::path(argv[2]).extension() != ".index") {
            use_attribute = true;
            attribute_path = argv[2];
            index_path = argv[3];
        } else {
            index_path = argv[2];
        }
    } else if (argc == 5) {
        use_attribute = true;
        attribute_path = argv[2];
        index_path = argv[3];
        num_threads = std::stoi(argv[4]);
    }

    std::cout << "[FAVOR] baseset_path=" << baseset_path
              << ", index_path=" << index_path
              << ", use_attribute=" << (use_attribute ? "true" : "false")
              << ", num_threads=" << num_threads << std::endl;
    if (use_attribute) {
        std::cout << "[FAVOR] attribute_path=" << attribute_path << std::endl;
    }

    BaseSet baseset;
    baseset.read_data(baseset_path);
    if (use_attribute) {
        baseset.get_attribute(attribute_path);
    } else {
        baseset.attribute_num = 0;
    }

    int dim = baseset.dim;
    int num = baseset.num;
    int M = 64;
    int ef_construction = 200;

    hnswlib::L2Space space(dim);
    favor::FAVOR<float> *alg_hnsw = new favor::FAVOR<float>(&space, num, M, ef_construction, baseset.attribute_num);

    std::cout << "begin building graph" << std::endl;
    auto build_start = std::chrono::steady_clock::now();
    ParallelFor(0, num, num_threads, [&](size_t i, size_t threadId) {
        float* attribute_ptr = nullptr;
        if (use_attribute && baseset.attribute_num > 0) {
            attribute_ptr = baseset.attribute + i * baseset.attribute_num;
        }
        alg_hnsw->addPoint(baseset.vectors.at(i).data(), baseset.vector_id[i], attribute_ptr);
    });
    auto build_end = std::chrono::steady_clock::now();
    std::cout << "finish building graph" << std::endl;
    auto build_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start).count();
    std::cout << "graph_build_time_ms=" << build_time_ms << std::endl;

    auto save_start = std::chrono::steady_clock::now();
    alg_hnsw->saveIndex(index_path);
    auto save_end = std::chrono::steady_clock::now();
    auto save_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(save_end - save_start).count();
    std::cout << "save_time_ms=" << save_time_ms << std::endl;

    std::cout << "save index in " << index_path << std::endl;

    delete alg_hnsw;
    return 0;
}
