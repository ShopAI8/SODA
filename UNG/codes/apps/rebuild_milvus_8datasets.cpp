#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include "storage.h"

#ifdef ENABLE_KNOWHERE_MILVUS_BASELINE
#include <knowhere/binaryset.h>
#include <knowhere/bitsetview.h>
#include <knowhere/comp/knowhere_config.h>
#include <knowhere/comp/index_param.h>
#include <knowhere/dataset.h>
#include <knowhere/index/index_factory.h>
#include <knowhere/version.h>
#endif

namespace fs = boost::filesystem;
namespace po = boost::program_options;

namespace {

const std::vector<std::string> kDefaultDatasets = {
    "Amazon",
    "BookReviews",
    "Genome",
    "Laion",
    "Music",
    "Reviews",
    "Tiktok",
    "VariousImg",
};

struct MilvusKnowhereCacheMeta {
   std::string index_kind;
   int64_t rows = 0;
   int64_t dim = 0;
   int nlist = 0;
   int nprobe = 0;
   double build_time_ms = -1.0;
   double train_time_ms = -1.0;
   double add_time_ms = -1.0;
   double index_sizeof_mb = 0.0;
   double serialized_index_size_mb = 0.0;
};

bool
save_milvus_knowhere_meta(const std::string& meta_path, const MilvusKnowhereCacheMeta& meta) {
   std::ofstream out(meta_path, std::ios::trunc);
   if (!out.is_open()) {
      return false;
   }
   out << std::fixed << std::setprecision(6);
   out << "index_kind=" << meta.index_kind << "\n";
   out << "rows=" << meta.rows << "\n";
   out << "dim=" << meta.dim << "\n";
   out << "nlist=" << meta.nlist << "\n";
   out << "nprobe=" << meta.nprobe << "\n";
   if (meta.build_time_ms >= 0.0) {
      out << "build_time_ms=" << meta.build_time_ms << "\n";
   }
   if (meta.train_time_ms >= 0.0) {
      out << "train_time_ms=" << meta.train_time_ms << "\n";
   }
   if (meta.add_time_ms >= 0.0) {
      out << "add_time_ms=" << meta.add_time_ms << "\n";
   }
   out << "index_sizeof_mb=" << meta.index_sizeof_mb << "\n";
   out << "serialized_index_size_mb=" << meta.serialized_index_size_mb << "\n";
   return out.good();
}

#ifdef ENABLE_KNOWHERE_MILVUS_BASELINE
bool
save_milvus_knowhere_index_file(const knowhere::Index<knowhere::IndexNode>& index,
                                const std::string& file_path,
                                uint64_t* serialized_size_bytes) {
   knowhere::BinarySet binset;
   if (index.Serialize(binset) != knowhere::Status::success || binset.binary_map_.empty()) {
      return false;
   }
   const auto& bin = binset.binary_map_.begin()->second;
   if (bin == nullptr || bin->data == nullptr || bin->size <= 0) {
      return false;
   }
   std::ofstream out(file_path, std::ios::binary | std::ios::trunc);
   if (!out.is_open()) {
      return false;
   }
   out.write(reinterpret_cast<const char*>(bin->data.get()), static_cast<std::streamsize>(bin->size));
   if (serialized_size_bytes != nullptr) {
      *serialized_size_bytes = static_cast<uint64_t>(bin->size);
   }
   return out.good();
}

struct BuildResult {
   MilvusKnowhereCacheMeta meta;
   bool ok = false;
   std::string error;
};

BuildResult
rebuild_ivf(const float* base_ptr, int64_t rows, int64_t dim, int num_threads,
            const fs::path& out_dir) {
   BuildResult result;
   result.meta.index_kind = "IVF_FLAT";
   result.meta.rows = rows;
   result.meta.dim = dim;
   result.meta.nlist = std::max<int>(1, std::min<int>(128, static_cast<int>(std::sqrt(static_cast<double>(rows)))));
   result.meta.nprobe = std::max<int>(1, std::min<int>(16, result.meta.nlist));

   auto idx_expected = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
       knowhere::IndexEnum::INDEX_FAISS_IVFFLAT,
       knowhere::Version::GetCurrentVersion().VersionNumber());
   if (!idx_expected.has_value()) {
      result.error = "failed to create INDEX_FAISS_IVFFLAT";
      return result;
   }

   auto index = idx_expected.value();
   const int kh_train_threads = 1;
   const int kh_add_threads = std::max<int>(1, std::min<int>(64, num_threads));

   knowhere::Json build_json;
   build_json[knowhere::meta::DIM] = dim;
   build_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
   build_json[knowhere::indexparam::NLIST] = result.meta.nlist;
   build_json[knowhere::indexparam::NPROBE] = result.meta.nprobe;
   build_json["use_elkan"] = false;

   knowhere::Json train_json = build_json;
   train_json[knowhere::meta::NUM_BUILD_THREAD] = kh_train_threads;
   knowhere::Json add_json = build_json;
   add_json[knowhere::meta::NUM_BUILD_THREAD] = kh_add_threads;

   const int64_t train_rows = std::min<int64_t>(rows, 30000);
   std::unique_ptr<float[]> owned_train(new float[static_cast<size_t>(train_rows) * static_cast<size_t>(dim)]);
   std::memcpy(owned_train.get(), base_ptr, sizeof(float) * static_cast<size_t>(train_rows) * static_cast<size_t>(dim));
   auto train_ds = knowhere::GenResultDataSet(train_rows, dim, std::move(owned_train));

   const auto build_begin = std::chrono::high_resolution_clock::now();
   const auto train_begin = std::chrono::high_resolution_clock::now();
   auto st = index.Train(train_ds, train_json, false);
   result.meta.train_time_ms = std::chrono::duration<double, std::milli>(
       std::chrono::high_resolution_clock::now() - train_begin).count();
   if (st != knowhere::Status::success) {
      result.error = "IVF train failed with status=" + std::to_string(static_cast<int>(st));
      return result;
   }

   std::unique_ptr<float[]> owned_base(new float[static_cast<size_t>(rows) * static_cast<size_t>(dim)]);
   std::memcpy(owned_base.get(), base_ptr, sizeof(float) * static_cast<size_t>(rows) * static_cast<size_t>(dim));
   auto add_ds = knowhere::GenResultDataSet(rows, dim, std::move(owned_base));
   const auto add_begin = std::chrono::high_resolution_clock::now();
   st = index.Add(add_ds, add_json, false);
   result.meta.add_time_ms = std::chrono::duration<double, std::milli>(
       std::chrono::high_resolution_clock::now() - add_begin).count();
   result.meta.build_time_ms = std::chrono::duration<double, std::milli>(
       std::chrono::high_resolution_clock::now() - build_begin).count();
   if (st != knowhere::Status::success) {
      result.error = "IVF add failed with status=" + std::to_string(static_cast<int>(st));
      return result;
   }

   result.meta.index_sizeof_mb = static_cast<double>(sizeof(index)) / (1024.0 * 1024.0);
   fs::create_directories(out_dir);
   uint64_t serialized_size_bytes = 0;
   if (!save_milvus_knowhere_index_file(index, (out_dir / "milvus_knowhere.index").string(),
                                        &serialized_size_bytes)) {
      result.error = "failed to save IVF serialized index";
      return result;
   }
   result.meta.serialized_index_size_mb = static_cast<double>(serialized_size_bytes) / (1024.0 * 1024.0);
   if (!save_milvus_knowhere_meta((out_dir / "milvus_knowhere.meta").string(), result.meta)) {
      result.error = "failed to save IVF meta";
      return result;
   }

   result.ok = true;
   return result;
}

BuildResult
rebuild_hnsw(const float* base_ptr, int64_t rows, int64_t dim, int num_threads,
             const fs::path& out_dir) {
   BuildResult result;
   result.meta.index_kind = "HNSW";
   result.meta.rows = rows;
   result.meta.dim = dim;
   result.meta.nlist = 32;
   result.meta.nprobe = 100;

   auto idx_expected = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
       knowhere::IndexEnum::INDEX_HNSW,
       knowhere::Version::GetCurrentVersion().VersionNumber());
   if (!idx_expected.has_value()) {
      result.error = "failed to create INDEX_HNSW";
      return result;
   }

   auto index = idx_expected.value();
   const int kh_add_threads = std::max<int>(1, std::min<int>(64, num_threads));

   knowhere::Json build_json;
   build_json[knowhere::meta::DIM] = dim;
   build_json[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
   build_json[knowhere::indexparam::HNSW_M] = result.meta.nlist;
   build_json[knowhere::indexparam::EFCONSTRUCTION] = 200;
   build_json[knowhere::indexparam::EF] = result.meta.nprobe;
   build_json[knowhere::meta::NUM_BUILD_THREAD] = kh_add_threads;

   std::unique_ptr<float[]> owned_base(new float[static_cast<size_t>(rows) * static_cast<size_t>(dim)]);
   std::memcpy(owned_base.get(), base_ptr, sizeof(float) * static_cast<size_t>(rows) * static_cast<size_t>(dim));
   auto ds = knowhere::GenResultDataSet(rows, dim, std::move(owned_base));

   const auto build_begin = std::chrono::high_resolution_clock::now();
   const auto st = index.Build(ds, build_json, false);
   result.meta.build_time_ms = std::chrono::duration<double, std::milli>(
       std::chrono::high_resolution_clock::now() - build_begin).count();
   if (st != knowhere::Status::success) {
      result.error = "HNSW build failed with status=" + std::to_string(static_cast<int>(st));
      return result;
   }

   result.meta.index_sizeof_mb = static_cast<double>(sizeof(index)) / (1024.0 * 1024.0);
   fs::create_directories(out_dir);
   uint64_t serialized_size_bytes = 0;
   if (!save_milvus_knowhere_index_file(index, (out_dir / "milvus_knowhere_hnsw.index").string(),
                                        &serialized_size_bytes)) {
      result.error = "failed to save HNSW serialized index";
      return result;
   }
   result.meta.serialized_index_size_mb = static_cast<double>(serialized_size_bytes) / (1024.0 * 1024.0);
   if (!save_milvus_knowhere_meta((out_dir / "milvus_knowhere_hnsw.meta").string(), result.meta)) {
      result.error = "failed to save HNSW meta";
      return result;
   }

   result.ok = true;
   return result;
}
#endif

bool
has_index_files(const fs::path& index_dir) {
   return fs::exists(index_dir / "index_files" / "vecs.bin") &&
          fs::exists(index_dir / "index_files" / "labels.txt");
}

std::map<std::string, std::string>
parse_kv_file_safe(const std::string& meta_path) {
   std::map<std::string, std::string> kvs;
   std::ifstream in(meta_path);
   if (!in.is_open()) {
      return kvs;
   }
   std::string line;
   while (std::getline(in, line)) {
      const auto pos = line.find('=');
      if (pos == std::string::npos) {
         continue;
      }
      kvs[line.substr(0, pos)] = line.substr(pos + 1);
   }
   return kvs;
}

int
resolve_build_threads(const fs::path& index_dir, int fallback_threads) {
   const auto meta = parse_kv_file_safe((index_dir / "index_files" / "meta").string());
   auto it = meta.find("build_num_threads");
   if (it == meta.end()) {
      return fallback_threads;
   }
   try {
      return std::max(1, std::stoi(it->second));
   } catch (...) {
      return fallback_threads;
   }
}

}  // namespace

int
main(int argc, char** argv) {
#ifndef ENABLE_KNOWHERE_MILVUS_BASELINE
   std::cerr << "ENABLE_KNOWHERE_MILVUS_BASELINE is disabled. This tool is unavailable." << std::endl;
   return 1;
#else
   std::string results_root = "/noraiddata/lijiakang/FilterVector/FilterVectorResults";
   std::vector<std::string> datasets = kDefaultDatasets;
   int num_threads = 60;

   try {
      po::options_description desc{"Arguments"};
      desc.add_options()
         ("help,h", "Print help")
         ("results_root", po::value<std::string>(&results_root)->default_value(results_root),
          "Root directory of FilterVectorResults")
         ("datasets", po::value<std::vector<std::string>>(&datasets)->multitoken(),
          "Optional dataset override")
         ("num_threads", po::value<int>(&num_threads)->default_value(num_threads),
          "Fallback build threads for Add/HNSW build when index meta has no build_num_threads");

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

   knowhere::KnowhereConfig::SetBuildThreadPoolSize(1);
   knowhere::KnowhereConfig::SetSearchThreadPoolSize(1);

   size_t total_indexes = 0;
   size_t ok_indexes = 0;

   for (const auto& dataset : datasets) {
      const fs::path dataset_index_root = fs::path(results_root) / dataset / "Index";
      if (!fs::exists(dataset_index_root)) {
         std::cerr << "[Skip] dataset index root not found: " << dataset_index_root.string() << std::endl;
         continue;
      }

      std::cout << "\n=== Dataset: " << dataset << " ===" << std::endl;
      for (fs::directory_iterator it(dataset_index_root), end; it != end; ++it) {
         if (!fs::is_directory(it->path()) || !has_index_files(it->path())) {
            continue;
         }
         total_indexes += 1;
         const fs::path index_dir = it->path();
         const fs::path index_files_dir = index_dir / "index_files";
         const fs::path bin_file = index_files_dir / "vecs.bin";
         const fs::path label_file = index_files_dir / "labels.txt";
         const fs::path ivf_dir = index_dir / "Milvus-IVF";
         const fs::path hnsw_dir = index_dir / "Milvus-HNSW";

         std::cout << "[Rebuild] " << index_dir.string() << std::endl;

         try {
            const int effective_threads = resolve_build_threads(index_dir, num_threads);
            auto base_storage = ANNS::create_storage("float", false);
            base_storage->load_from_file(bin_file.string(), label_file.string());
            const int64_t rows = static_cast<int64_t>(base_storage->get_num_points());
            const int64_t dim = static_cast<int64_t>(base_storage->get_dim());
            const float* base_ptr = reinterpret_cast<const float*>(base_storage->get_vector(0));
            if (base_ptr == nullptr || rows <= 0 || dim <= 0) {
               throw std::runtime_error("invalid base storage");
            }
            std::cout << "  build_threads=" << effective_threads
                      << " (aligned with index_files/meta build_num_threads when present)" << std::endl;

            const auto ivf = rebuild_ivf(base_ptr, rows, dim, effective_threads, ivf_dir);
            if (!ivf.ok) {
               throw std::runtime_error("IVF rebuild failed: " + ivf.error);
            }
            std::cout << "  IVF: total=" << ivf.meta.build_time_ms
                      << " ms, train=" << ivf.meta.train_time_ms
                      << " ms, add=" << ivf.meta.add_time_ms
                      << " ms, sizeof=" << ivf.meta.index_sizeof_mb
                      << " MB, serialized=" << ivf.meta.serialized_index_size_mb
                      << " MB" << std::endl;

            const auto hnsw = rebuild_hnsw(base_ptr, rows, dim, effective_threads, hnsw_dir);
            if (!hnsw.ok) {
               throw std::runtime_error("HNSW rebuild failed: " + hnsw.error);
            }
            std::cout << "  HNSW: total=" << hnsw.meta.build_time_ms
                      << " ms, sizeof=" << hnsw.meta.index_sizeof_mb
                      << " MB, serialized=" << hnsw.meta.serialized_index_size_mb
                      << " MB" << std::endl;

            ok_indexes += 1;
         } catch (const std::exception& ex) {
            std::cerr << "  Failed: " << ex.what() << std::endl;
         }
      }
   }

   std::cout << "\nFinished. success=" << ok_indexes
             << ", total=" << total_indexes << std::endl;
   return (ok_indexes == total_indexes) ? 0 : 2;
#endif
}
