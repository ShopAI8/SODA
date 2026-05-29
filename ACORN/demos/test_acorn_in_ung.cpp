#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <stdexcept> // For std::runtime_error
#include <unordered_set>

// FAISS and system headers
#include <sys/time.h>
#include <sys/stat.h>
#include <unistd.h>
#include "../faiss/IndexACORN.h"
#include "../faiss/index_io.h"
#include "../faiss/utils/distances.h"
#include "utils.cpp"

// Function to parse command-line arguments
std::map<std::string, std::string> parse_arguments(int argc, char *argv[])
{
   std::map<std::string, std::string> args;
   for (int i = 1; i < argc; ++i)
   {
      std::string arg = argv[i];
      if (arg.rfind("--", 0) == 0)
      {
         std::string key = arg.substr(2);
         if (i + 1 < argc && (std::string(argv[i + 1]).rfind("--", 0) != 0))
         {
            args[key] = argv[++i];
         }
         else
         {
            args[key] = "1"; // Flag argument
         }
      }
   }
   return args;
}

// Function to print usage instructions
void print_usage(const char *prog_name)
{
   std::cerr << "Usage: " << prog_name << " [arguments]\n\n"
             << "Required arguments:\n"
             << "  --dataset <string>         Dataset name, for example sift1m\n"
             << "  --base_path <path>         Base vector data directory\n"
             << "  --base_label_path <path>   Base vector attribute-label directory\n"
             << "  --query_vec_path <path>    Query vector file path\n"
             << "  --query_attr_path <path>   Query attribute file directory\n"
             << "  --output_path <path>       Result output file path\n"
             << "  --N <int>                  Number of database vectors\n"
             << "  --M <int>                  Number of neighbors in the ACORN graph\n"
             << "  --M_beta <int>             Number of neighbors in the ACORN compression layer\n"
             << "  --gamma <int>              Number of ACORN attribute partitions\n"
             << "  --efs <int>                ACORN efSearch parameter\n"
             << "  --k <int>                  Number of nearest neighbors to retrieve\n"
             << "  --threads <int>            Number of threads to use\n\n"
             << "Optional arguments:\n"
             << "  --compute_recall           Enable recall computation, which may be time-consuming\n"
             << std::endl;
}

// Function to compute ground truth with attribute filtering
std::vector<std::vector<faiss::idx_t>> compute_ground_truth(
    size_t nq, size_t N, size_t d, int k,
    const std::vector<float> &xq, const std::vector<float> &xb,
    const std::vector<char> &filter_ids_map)
{
   std::vector<std::vector<faiss::idx_t>> ground_truth(nq);
#pragma omp parallel for
   for (int i = 0; i < nq; ++i)
   {
      const float *query_vector = xq.data() + i * d;
      std::vector<std::pair<float, faiss::idx_t>> distances;

      for (int j = 0; j < N; ++j)
      {
         if (filter_ids_map[i * N + j])
         {
            const float *base_vector = xb.data() + j * d;
            float dist = faiss::fvec_L2sqr(query_vector, base_vector, d);
            distances.push_back({dist, (faiss::idx_t)j});
         }
      }
      std::sort(distances.begin(), distances.end());
      for (int m = 0; m < k && m < distances.size(); ++m)
      {
         ground_truth[i].push_back(distances[m].second);
      }
   }
   return ground_truth;
}

// Function to calculate per-query recall and return a vector of recalls
std::vector<float> calculate_per_query_recall(
    const std::vector<faiss::idx_t> &results_labels,
    const std::vector<std::vector<faiss::idx_t>> &ground_truth,
    size_t nq, int k)
{
   std::vector<float> per_query_recalls;
   per_query_recalls.reserve(nq);

   for (int i = 0; i < nq; ++i)
   {
      const auto &gt_set = ground_truth[i];
      size_t gt_size = gt_set.size();

      if (gt_size == 0)
      {
         per_query_recalls.push_back(1.0f);
         continue;
      }

      std::unordered_set<faiss::idx_t> gt_unordered_set(gt_set.begin(), gt_set.end());
      long long found_count = 0;

      for (int j = 0; j < k; ++j)
      {
         faiss::idx_t result_id = results_labels[i * k + j];
         if (gt_unordered_set.count(result_id))
         {
            found_count++;
         }
      }
      per_query_recalls.push_back((float)found_count / gt_size);
   }
   return per_query_recalls;
}

int main(int argc, char *argv[])
{
   // --- 0. Initial Setup ---
   double t_start = elapsed();
   std::cout << "=======================================\n"
             << "===   ACORN Search Tool for UNG   ===\n"
             << "=======================================\n";

   // --- 1. Argument Parsing ---
   auto args = parse_arguments(argc, argv);
   std::vector<std::string> required_args = {
       "dataset", "base_path", "base_label_path", "query_vec_path",
       "query_attr_path", "output_path", "N", "M", "M_beta", "gamma", "efs", "k", "threads"};

   for (const auto &key : required_args)
   {
      if (args.find(key) == args.end())
      {
         std::cerr << "Error: Missing required argument --" << key << std::endl;
         print_usage(argv[0]);
         return 1;
      }
   }

   // --- 2. Load Parameters ---
   std::string dataset = args["dataset"];
   std::string base_path = args["base_path"];
   std::string base_label_path = args["base_label_path"];
   std::string query_vec_path = args["query_vec_path"];
   std::string query_attr_path = args["query_attr_path"];
   std::string output_path = args["output_path"];
   size_t N = std::stoul(args["N"]);
   int M = std::stoi(args["M"]);
   int M_beta = std::stoi(args["M_beta"]);
   int gamma = std::stoi(args["gamma"]);
   int efs = std::stoi(args["efs"]);
   int k = std::stoi(args["k"]);
   int nthreads = std::stoi(args["threads"]);
   bool compute_recall = args.count("compute_recall") > 0;

   omp_set_num_threads(nthreads);
   std::cout << "Argument parsing completed. Using " << nthreads << " threads." << std::endl;
   if (compute_recall)
   {
      std::cout << "Recall computation is enabled." << std::endl;
   }

   // --- 3. Load Data & Attributes ---
   // Note: We use std::vector for safer memory management.
   size_t d = 0;
   size_t nq = 0;
   std::vector<float> xb;
   std::vector<float> xq;
   std::vector<std::vector<int>> metadata;
   std::vector<std::vector<int>> aq;

   try
   {
      // Load base vectors
      std::cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t_start << " s] Loading database vectors..." << std::endl;
      size_t d_base, nb;
      std::string base_filename = base_path + "/" + dataset + "_base.fvecs";
      float *xb_raw = fvecs_read(base_filename.c_str(), &d_base, &nb);
      if (!xb_raw)
         throw std::runtime_error("Unable to read base vector file: " + base_filename);
      d = d_base;
      xb.assign(xb_raw, xb_raw + nb * d);
      delete[] xb_raw; // Immediately free raw pointer after copy
      std::cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t_start << " s] Loaded " << nb << " database vectors (dimension: " << d << ")." << std::endl;

      // Load query vectors
      std::cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t_start << " s] Loading query vectors..." << std::endl;
      size_t d_query;
      float *xq_raw = fvecs_read(query_vec_path.c_str(), &d_query, &nq);
      if (!xq_raw)
         throw std::runtime_error("Unable to read query vector file: " + query_vec_path);
      if (d != d_query)
         throw std::runtime_error("Base vectors and query vectors have mismatched dimensions.");
      xq.assign(xq_raw, xq_raw + nq * d);
      delete[] xq_raw;
      std::cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t_start << " s] Loaded " << nq << " query vectors (dimension: " << d << ")." << std::endl;

      // Load base attributes
      std::cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t_start << " s] Loading database attributes..." << std::endl;
      metadata = load_ab_muti(dataset, gamma, "rand", N, base_label_path);
      metadata.resize(N);
      for (auto &vec : metadata)
      {
         std::sort(vec.begin(), vec.end());
      }
      std::cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t_start << " s] Loaded attributes for " << metadata.size() << " database entries." << std::endl;

      // Load query attributes
      std::cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t_start << " s] Loading query attributes..." << std::endl;
      aq = load_txt_to_vector_multi<int>(query_attr_path);
      for (auto &vec : aq)
      {
         std::sort(vec.begin(), vec.end());
      }
      std::cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t_start << " s] Loaded attributes for " << aq.size() << " queries." << std::endl;

      if (nq != aq.size())
      {
         throw std::runtime_error("The number of query vectors does not match the number of query attributes.");
      }
   }
   catch (const std::exception &e)
   {
      std::cerr << "Error while loading data: " << e.what() << std::endl;
      return 1;
   }

   // --- 4. Build ACORN Index ---
   std::cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t_start << " s] Creating ACORN index (M=" << M << ", gamma=" << gamma << ")..." << std::endl;
   faiss::IndexACORNFlat hybrid_index(d, M, gamma, metadata, M_beta);
   double t_build_0 = elapsed();
   hybrid_index.add(N, xb.data());
   double t_build_1 = elapsed();
   std::cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t_start << " s] Added " << N << " vectors to the index successfully. Build time: " << t_build_1 - t_build_0 << " s." << std::endl;
   hybrid_index.printStats(false);

   // --- 5. Perform Filtered Search ---
   std::cout << "==================== Starting Search ====================\n";
   std::cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t_start << " s] Searching " << k << " nearest neighbors for " << nq << " queries (efs=" << efs << ")..." << std::endl;

   hybrid_index.acorn.efSearch = efs;

   std::vector<faiss::idx_t> result_labels(k * nq, -1);
   std::vector<float> result_dists(k * nq, -1.0f);

   std::cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t_start << " s] Creating attribute filter..." << std::endl;
   double t_filter_0 = elapsed();
   std::vector<char> filter_ids_map(nq * N);
#pragma omp parallel for
   for (int i_q = 0; i_q < nq; i_q++)
   {
      for (int i_b = 0; i_b < N; i_b++)
      {
         const auto &query_attrs = aq[i_q];
         const auto &data_attrs = metadata[i_b];
         filter_ids_map[i_q * N + i_b] = std::includes(
             data_attrs.begin(), data_attrs.end(),
             query_attrs.begin(), query_attrs.end());
      }
   }
   double t_filter_1 = elapsed();
   std::cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t_start << " s] Attribute filter created. Time: " << t_filter_1 - t_filter_0 << " s" << std::endl;

   double t_search_0 = elapsed();
   hybrid_index.search(nq, xq.data(), k, result_dists.data(), result_labels.data(), filter_ids_map.data(), nullptr, nullptr, nullptr, true);
   double t_search_1 = elapsed();

   double search_time = t_search_1 - t_search_0;
   double qps = (search_time > 0) ? (nq / search_time) : 0;
   std::cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t_start << " s] Search completed. Total time: " << search_time << " s, QPS: " << std::fixed << std::setprecision(2) << qps << std::endl;

   // --- 6. Calculate Recall (Optional) ---
   if (compute_recall)
   {
      std::cout << "==================== Computing Recall ====================\n";
      std::cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t_start << " s] Computing ground truth nearest neighbors. This may take a long time..." << std::endl;
      double t_gt_0 = elapsed();
      auto ground_truth = compute_ground_truth(nq, N, d, k, xq, xb, filter_ids_map);
      double t_gt_1 = elapsed();
      std::cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t_start << " s] Ground truth computation completed. Time: " << t_gt_1 - t_gt_0 << " s\n";

      // 1. Calculate recall for each query individually
      auto per_query_recalls = calculate_per_query_recall(result_labels, ground_truth, nq, k);

      double total_recall_sum = 0.0;
      int queries_with_gt = 0;

      std::cout << "\n--- Per-query Recall@" << k << " ---\n";
      std::cout << std::fixed << std::setprecision(4);
      for (size_t i = 0; i < per_query_recalls.size(); ++i)
      {
         std::cout << "Query " << std::setw(4) << i << ": " << per_query_recalls[i] << std::endl;
         if (!ground_truth[i].empty())
         {
            total_recall_sum += per_query_recalls[i];
            queries_with_gt++;
         }
      }

      // 2. Calculate the average of the per-query recalls
      double average_recall = (queries_with_gt > 0) ? total_recall_sum / queries_with_gt : 0.0;

      std::cout << "\n>>> Macro Average Recall@" << k << ": " << average_recall << "\n\n";
   }

   // --- 7. Save Results ---
   std::cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t_start << " s] Writing results, including full vectors, to " << output_path << "...\n";
   std::ofstream out_file(output_path);
   if (!out_file.is_open())
   {
      std::cerr << "Error: Unable to open output file " << output_path << std::endl;
      return 1;
   }
   out_file << std::fixed << std::setprecision(6);
   for (int i = 0; i < nq; ++i)
   {
      out_file << "Query: " << i << "\n";
      for (int j = 0; j < k; ++j)
      {
         faiss::idx_t neighbor_id = result_labels[i * k + j];
         if (neighbor_id < 0)
         {
            out_file << "NeighborID: -1 Vector: \n";
            continue;
         }
         out_file << "NeighborID: " << neighbor_id << " Vector:";
         const float *vector_data = xb.data() + neighbor_id * d;
         for (int dim = 0; dim < d; ++dim)
         {
            out_file << " " << vector_data[dim];
         }
         out_file << "\n";
      }
      out_file << "---\n";
   }
   out_file.close();
   std::cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t_start << " s] Result writing completed.\n";

   // --- 8. Cleanup & Exit ---
   // No manual delete[] needed for xb and xq thanks to std::vector!
   std::cout << "[" << std::fixed << std::setprecision(3) << elapsed() - t_start << " s] -----   Task completed   -----\n";
   return 0;
}
