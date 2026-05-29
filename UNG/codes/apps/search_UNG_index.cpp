#include <chrono>
#include <fstream>
#include <numeric>
#include <iostream>
#include <bitset>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include "uni_nav_graph.h"
#include "utils.h"
#include <roaring/roaring.h>
#include <roaring/roaring.hh>
#include <faiss_navix/IndexHNSW.h>
#include <faiss_navix/index_io.h>

namespace po = boost::program_options;
namespace fs = boost::filesystem;

// Helper function: compute recall for a single query, including distance-tie handling
float calculate_single_query_recall(const std::pair<ANNS::IdxType, float> *gt,
                                    const std::pair<ANNS::IdxType, float> *results,
                                    ANNS::IdxType K)
{
   // 1. Get the distance threshold from the K-th GT result. GT is sorted and pair.second is the distance.
   // Handle boundary cases where GT contains fewer than K valid results.
   float gt_threshold = 0.0f;
   if (K > 0 && gt[K - 1].first != -1) {
       gt_threshold = gt[K - 1].second;
   } else {
       // If GT contains fewer than K results, use the last valid entry
       for(int i = K - 1; i >= 0; i--) {
           if(gt[i].first != -1) {
               gt_threshold = gt[i].second;
               break;
           }
       }
   }
   
   // Floating-point tolerance
   float epsilon = 1e-6; 

   std::unordered_set<ANNS::IdxType> gt_set;
   for (int i = 0; i < K; ++i)
   {
      if (gt[i].first != -1)
      {
         gt_set.insert(gt[i].first);
      }
   }
   
   if (gt_set.empty()) return 1.0f; // Empty GT is treated as 100% recall by convention

   int correct = 0;
   for (int i = 0; i < K; ++i)
   {
      if (results[i].first == -1) continue;

      // Condition 1: ID hit
      if (gt_set.count(results[i].first))
      {
         correct++;
      }
      // Condition 2: ID miss but distance is within the accepted tie threshold
      // Ensure that results[i].second is valid before applying this check
      else if (results[i].second <= gt_threshold + epsilon) 
      {
         correct++;
      }
   }

   return static_cast<float>(correct) / gt_set.size();
}


int main(int argc, char **argv)
{
   std::string data_type, dist_fn, scenario;
   std::string base_bin_file, query_bin_file, base_label_file, query_label_file, gt_file, index_path_prefix, result_path_prefix, selector_modle_prefix, query_group_id_file;
   std::string acorn_index_path, acorn_1_index_path, navix_index_path, algo_choice_csv_path;
   ANNS::IdxType K, num_entry_points;
   std::vector<ANNS::IdxType> Lsearch_list;
   uint32_t num_threads;
   bool is_new_method = false;                                 // true: use new method
   bool is_new_trie_method = false, is_rec_more_start = false; // false: original UNG Trie method; true: recursive method; false: default root entry
   bool is_ung_more_entry = false;                             // false: original UNG entry-point selection; true: use additional entry points
   // bool is_bfs_filter = true;                                  // true: original ACORN; false: improved variant
   int baseline_alg = 0; // 0/1/8=UNG, 2/3/4/6=ACORN, 5=pre-filter, 7=NaviX, 9=Milvus-IVF, 10=Milvus-HNSW (Knowhere)
   int num_repeats = 1;                                        // Default to one repeat
   int routing_mode = 0;                                      // 0: auto, 1: UNG (nT=false), 2: UNG-nTtrue, 3: ACORN
   int lsearch_start, lsearch_step;
   int efs_start, efs_step_slow, efs_step_fast, lsearch_threshold;
   std::string dataset; 
   std::string ung_distance_mode = "exact";
   bool optimize_standalone_prefilter = false; // Disable standalone pre-filter optimization by default; enable it for large queries when needed

   try
   {
      po::options_description desc{"Arguments"};
      desc.add_options()("help,h", "Print information on arguments");
      desc.add_options()("dataset", po::value<std::string>(&dataset)->required(),
                         "dataset");
      desc.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                         "data type <int8/uint8/float>");
      desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                         "distance function <L2/IP/cosine>");
      desc.add_options()("base_bin_file", po::value<std::string>(&base_bin_file)->required(),
                         "File containing the base vectors in binary format");
      desc.add_options()("query_bin_file", po::value<std::string>(&query_bin_file)->required(),
                         "File containing the query vectors in binary format");
      desc.add_options()("base_label_file", po::value<std::string>(&base_label_file)->default_value(""),
                         "Base label file in txt format");
      desc.add_options()("query_label_file", po::value<std::string>(&query_label_file)->default_value(""),
                         "Query label file in txt format");
      desc.add_options()("gt_file", po::value<std::string>(&gt_file)->required(),
                         "Filename for the computed ground truth in binary format");
      desc.add_options()("K", po::value<ANNS::IdxType>(&K)->required(),
                         "Number of ground truth nearest neighbors to compute");
      desc.add_options()("num_threads", po::value<uint32_t>(&num_threads)->default_value(ANNS::default_paras::NUM_THREADS),
                         "Number of threads to use");
      desc.add_options()("result_path_prefix", po::value<std::string>(&result_path_prefix)->required(),
                         "Path to save the querying result file");
      desc.add_options()("selector_modle_prefix", po::value<std::string>(&selector_modle_prefix)->required(),
                         "Path to selector_modle_prefix");
      desc.add_options()("query_group_id_file", po::value<std::string>(&query_group_id_file)->required(),
                         "query_group_id_file");
      desc.add_options()("acorn_index_path", po::value<std::string>(&acorn_index_path)->default_value(""),
                         "acorn_index_path");
      desc.add_options()("acorn_1_index_path", po::value<std::string>(&acorn_1_index_path)->default_value(""),
                         "acorn_1_index_path");

      // graph search parameters
      desc.add_options()("scenario", po::value<std::string>(&scenario)->default_value("containment"),
                         "Scenario for building UniNavGraph, <equality/containment/overlap/nofilter>");
      desc.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                         "Prefix of the path to load the index");
      desc.add_options()("num_entry_points", po::value<ANNS::IdxType>(&num_entry_points)->default_value(ANNS::default_paras::NUM_ENTRY_POINTS),
                         "Number of entry points in each entry group");
      desc.add_options()("Lsearch", po::value<std::vector<ANNS::IdxType>>(&Lsearch_list)->multitoken()->required(),
                         "Number of candidates to search in the graph");
      desc.add_options()("is_new_method", po::value<bool>(&is_new_method)->required(),
                         "is_new_method");
      desc.add_options()("is_new_trie_method", po::value<bool>(&is_new_trie_method)->required(),
                         "is_new_trie_method");
      desc.add_options()("is_rec_more_start", po::value<bool>(&is_rec_more_start)->required(),
                         "is_rec_more_start");
      // desc.add_options()("is_bfs_filter", po::value<bool>(&is_bfs_filter)->default_value(true), "Whether to use BFS filter in ACORN");
      desc.add_options()("baseline_alg", po::value<int>(&baseline_alg)->default_value(0), "Algorithm selector: 0/1/8=UNG family, 2/3/4/6=ACORN family, 5=pre-filter, 7=NaviX, 9=Milvus-IVF, 10=Milvus-HNSW");
      desc.add_options()("num_repeats", po::value<int>(&num_repeats)->default_value(1),
                         "Number of repeats for each Lsearch value");
      desc.add_options()("routing_mode", po::value<int>(&routing_mode)->required(),
                         "0: auto, 1: SmartRoute, 2: FastSmartRoute, 3: FastSmartRoute+, 5: SmartRoute+, 6: SmartRoute++, 7: SmartRoute+++");
      desc.add_options()("algo_choice_csv", po::value<std::string>(&algo_choice_csv_path)->default_value(""),
                         "Optional CSV path for per-query algorithm override. Format: QueryID,Algo_Choice");
      desc.add_options()("lsearch_start", po::value<int>(&lsearch_start)->required(), "Lsearch start value");
      desc.add_options()("lsearch_step", po::value<int>(&lsearch_step)->required(), "Lsearch step value");
      desc.add_options()("efs_start", po::value<int>(&efs_start)->required(), "ACORN efs start value");
      desc.add_options()("efs_step_slow", po::value<int>(&efs_step_slow)->required(), "ACORN efs step value");
      desc.add_options()("efs_step_fast", po::value<int>(&efs_step_fast)->required(), "ACORN efs step value");
      desc.add_options()("lsearch_threshold", po::value<int>(&lsearch_threshold)->required(), "lsearch_threshold");
      desc.add_options()("ung_distance_mode", po::value<std::string>(&ung_distance_mode)->default_value("exact"),
                         "UNG distance mode: exact (rabitq is disabled unless UNG_ENABLE_RABITQ=ON)");

      // NaviX
      desc.add_options()("navix_index_path", po::value<std::string>(&navix_index_path)->default_value(""), "Path to NaviX index");

      desc.add_options()("optimize_standalone_prefilter", po::value<bool>(&optimize_standalone_prefilter)->default_value(false),
                   "Whether to fully optimize pre-filter when running as standalone baseline");



      po::variables_map vm;
      po::store(po::parse_command_line(argc, argv, desc), vm);
      if (vm.count("help"))
      {
         std::cout << desc;
         return 0;
      }
      po::notify(vm);
   }
   catch (const std::exception &ex)
   {
      std::cerr << ex.what() << std::endl;
      return -1;
   }

   // check scenario
   if (scenario != "containment" && scenario != "equality" && scenario != "overlap")
   {
      std::cerr << "Invalid scenario: " << scenario << std::endl;
      return -1;
   }
#if !UNG_ENABLE_RABITQ
   if (ung_distance_mode == "rabitq")
   {
      std::cerr << "[RabitQ] RabitQ support is disabled at compile time. Falling back to exact UNG distance mode." << std::endl;
      ung_distance_mode = "exact";
   }
#endif

   // load query data
   std::shared_ptr<ANNS::IStorage> query_storage = ANNS::create_storage(data_type);
   query_storage->load_from_file(query_bin_file, query_label_file);

   // load index
   ANNS::UniNavGraph index(query_storage->get_num_points());
   index.load(index_path_prefix, selector_modle_prefix, data_type, acorn_index_path, acorn_1_index_path,dataset);
   index.set_ung_distance_mode(ung_distance_mode);
   index.prepare_rabitq_query_contexts(query_storage, query_bin_file);

   // `index.load(...)` already loads `vector_attr_graph` and builds the
   // vector-level inverted index when the bipartite graph file exists.
   // Keep only the group-level inverted index construction here.
   index.build_group_inverted_indices();


   // Naxiv
   faiss_navix::IndexHNSWFlat* navix_index = nullptr;
   if (routing_mode == 0) { // Load NaviX on demand in automatic routing mode
      if (!navix_index_path.empty() && fs::exists(navix_index_path)) {
         std::cout << "[SmartRoute] Loading NaviX index from: " << navix_index_path << std::endl;
         faiss_navix::Index* raw_navix = faiss_navix::read_index(navix_index_path.c_str());
         navix_index = dynamic_cast<faiss_navix::IndexHNSWFlat*>(raw_navix);
         if (!navix_index) {
            std::cerr << "ERROR: Failed to cast loaded NaviX index to faiss_navix::IndexHNSWFlat" << std::endl;
            delete raw_navix;
         }
      } else {
         std::cout << "[Warning] NaviX index path is empty or does not exist. NaviX routing will fail." << std::endl;
      }
   }

   // Load the query source-group ID file
   std::vector<ANNS::IdxType> true_query_group_ids;
   std::ifstream source_group_file(query_group_id_file);
   if (source_group_file.is_open())
   {
      ANNS::IdxType group_id;
      while (source_group_file >> group_id)
      {
         true_query_group_ids.push_back(group_id);
      }
      source_group_file.close();
      std::cout << "Successfully loaded source-group IDs for " << true_query_group_ids.size() << " queries." << std::endl;
   }
   else // The program can continue without this file, but the optimization will be unavailable
   {
      std::cerr << "Warning: Query source-group ID file not found: " << query_group_id_file << std::endl;
   }

   // preparation
   auto num_queries = query_storage->get_num_points();
   std::shared_ptr<ANNS::DistanceHandler> distance_handler = ANNS::get_distance_handler(data_type, dist_fn);
   auto gt = new std::pair<ANNS::IdxType, float>[num_queries * K];
   ANNS::load_gt_file(gt_file, gt, num_queries, K);
   auto results = new std::pair<ANNS::IdxType, float>[num_queries * K];
   std::vector<int> query_algo_choices = index.load_query_algo_choices_from_csv(algo_choice_csv_path, num_queries);

   // ==================== [SmartRoute+/SmartRoute+++ Global Preprocessing, executed once] ====================
   std::vector<ANNS::QueryStats> global_pred_stats(num_queries);
   double total_global_pred_time = 0.0;
   double total_global_sort_time = 0.0;
   std::vector<int> final_global_choices = query_algo_choices;
   std::vector<int> sorted_query_ids(num_queries);
   std::iota(sorted_query_ids.begin(), sorted_query_ids.end(), 0);

   if (routing_mode == 5 || routing_mode == 7) {
       // --- Run multiple trials and keep the shortest global prediction time ---
       double min_pred_time = std::numeric_limits<double>::max();
       std::vector<int> best_choices;
       std::vector<ANNS::QueryStats> best_stats;
       
       int num_trials = 3; // Number of timing trials
       for (int trial = 0; trial < num_trials; ++trial) {
           auto global_pred_start = std::chrono::high_resolution_clock::now();
           std::vector<ANNS::QueryStats> trial_stats(num_queries);
           auto trial_choices = index.global_predict_algo_choices(
               query_storage, routing_mode, query_algo_choices, num_threads, trial_stats
           );
           double trial_time = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - global_pred_start).count();
           
           std::cout << "[SmartRoute" << (routing_mode == 7 ? "+++" : "+") << "] Trial " << trial + 1 << " Prediction Time: " << trial_time << " ms" << std::endl;
           
           if (trial_time < min_pred_time) {
               min_pred_time = trial_time;
               best_choices = std::move(trial_choices);
               best_stats = std::move(trial_stats);
           }
       }
       
       // Adopt the best result
       final_global_choices = std::move(best_choices);
       global_pred_stats = std::move(best_stats);
       total_global_pred_time = min_pred_time;
       std::cout << "\n[SmartRoute" << (routing_mode == 7 ? "+++" : "+") << "] Best Global Prediction Time: " << total_global_pred_time << " ms" << std::endl;

       // 2. Global cache-friendly ordering
       auto global_sort_start = std::chrono::high_resolution_clock::now();
       sorted_query_ids = index.get_sorted_query_ids(query_storage, final_global_choices, routing_mode);
       total_global_sort_time = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - global_sort_start).count();
       std::cout << "[SmartRoute" << (routing_mode == 7 ? "+++" : "+") << "] Global Sort Time: " << total_global_sort_time << " ms" << std::endl;

       double avg_sort_ms = total_global_sort_time / num_queries;
       for (int i = 0; i < (int)num_queries; ++i) {
           global_pred_stats[i].global_sort_time_ms = avg_sort_ms;
       }
       std::cout << "[SmartRoute" << (routing_mode == 7 ? "+++" : "+") << "] Total Preprocessing Time: " << (total_global_pred_time + total_global_sort_time) << " ms" << std::endl;
   }
   // Warm-up selector
   std::cout << "\n--- Starting Warm-up Phase ---" << std::endl;
   index.warmup_selectors(num_threads);
   std::cout << "--- Warm-up Finished ---"<< std::endl;

   if (baseline_alg == 8 || routing_mode == 1 || routing_mode == 5 || routing_mode == 6 || routing_mode == 7) {
    index.skip_els_filter = true;
    std::cout << "[UNG+] Mode Enabled: ELS filtering will be skipped." << std::endl;
   }

   // init query stats
   std::vector<std::vector<std::vector<ANNS::QueryStats>>> query_stats(num_repeats, std::vector<std::vector<ANNS::QueryStats>>(Lsearch_list.size(), std::vector<ANNS::QueryStats>(num_queries))); //(repeat,Lsearch,queryID)

   // Struct used to store detailed timing for each run
   struct SearchTimeLog
   {
      int repeat;
      ANNS::IdxType l_search;
      int efs;
      double time_ms;
      float avg_recall;

      // ELS-related timing fields
      double els_trie_avg;
      double els_sort_avg;
      double els_filter_avg;
      double els_total_avg;

      double mask_gen_avg;
      double global_sort_avg;
   };
   std::vector<SearchTimeLog> detailed_times;                      // Store all detailed timing records
   std::map<ANNS::IdxType, std::vector<double>> time_per_lsearch;  // Group repeat timings by Lsearch for later averaging
   std::map<ANNS::IdxType, std::vector<float>> recall_per_lsearch; // Store recall values for each Lsearch
   std::map<ANNS::IdxType, std::vector<int>> efs_per_lsearch;      // Store efs values for each Lsearch

   for (int repeat = 0; repeat < num_repeats; ++repeat)
   {
      std::cout << "\n=== Repeat " << (repeat + 1) << "/" << num_repeats << " ===" << std::endl;

      // search
      std::vector<float> all_cmps, all_qpss, all_recalls;
      std::vector<float> all_time_ms, all_flag_time, all_bitmap_time, all_entry_points, all_lng_descendants, all_entry_group_coverage;
      std::vector<float> all_is_global_search; // Reserved for global-search ratio statistics if needed

      std::cout << "Start querying ..." << std::endl;
      for (int LsearchId = 0; LsearchId < Lsearch_list.size(); LsearchId++)
      {
         ANNS::IdxType current_Lsearch = Lsearch_list[LsearchId];
         std::vector<float> num_cmps(num_queries);

         // --- 1. Assemble the query queue for this run and dispatch global preprocessing data ---
         std::queue<int> task_queue;
         for (int id : sorted_query_ids) {
             task_queue.push(id);
         }

         if (routing_mode == 5 || routing_mode == 7) {
             for (int i = 0; i < (int)num_queries; ++i) {
                 query_stats[repeat][LsearchId][i].mask_gen_time_ms = global_pred_stats[i].mask_gen_time_ms;
                 query_stats[repeat][LsearchId][i].route_pred_time_ms = global_pred_stats[i].route_pred_time_ms;
                 query_stats[repeat][LsearchId][i].global_sort_time_ms = global_pred_stats[i].global_sort_time_ms;
                 query_stats[repeat][LsearchId][i].exact_cand_size = global_pred_stats[i].exact_cand_size;
                 query_stats[repeat][LsearchId][i].global_p_pass = global_pred_stats[i].global_p_pass;
             }
         }

         // --- 2. Time and execute the search ---
         auto start_time = std::chrono::high_resolution_clock::now();
         if (!is_new_method)
         {
            // index.search(...);
         }
         else
         {
             index.search_hybrid(query_storage, distance_handler, num_threads, current_Lsearch,
                                num_entry_points, scenario, K, results, num_cmps, query_stats[repeat][LsearchId],is_new_trie_method, is_rec_more_start, is_ung_more_entry, lsearch_start, lsearch_step, efs_start, efs_step_slow,efs_step_fast,lsearch_threshold,routing_mode, baseline_alg ,navix_index, true_query_group_ids, final_global_choices, task_queue, optimize_standalone_prefilter);
         }
         double pure_search_time = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time).count();
         
         // --- 3. Apply batch-level timing compensation ---
         double time_cost = pure_search_time;
         if (routing_mode == 5 || routing_mode == 7) {
             time_cost += (total_global_pred_time + total_global_sort_time);
         }

         // --- 4. Compute recall for each individual query ---
         for (int i = 0; i < num_queries; ++i)
            query_stats[repeat][LsearchId][i].recall = calculate_single_query_recall(gt + i * K, results + i * K, K);

         // --- 5. Compute the average recall for the current Lsearch batch ---
         double total_recall_for_batch = 0.0;
         int total_efs_for_batch = 0;
         double total_ndc_for_batch = 0.0;
         double sum_trie = 0.0;
         double sum_sort = 0.0;
         double sum_filter = 0.0;
         double sum_total = 0.0;
         double sum_mask_gen = 0.0;
         double sum_global_sort = 0.0;

         for (int i = 0; i < num_queries; ++i){
            total_recall_for_batch += query_stats[repeat][LsearchId][i].recall;
            total_efs_for_batch += query_stats[repeat][LsearchId][i].acorn_efs_used;
            total_ndc_for_batch += query_stats[repeat][LsearchId][i].num_distance_calcs;

            // Accumulate timings for each ELS stage
            const auto &s = query_stats[repeat][LsearchId][i];
            sum_trie += s.els_trie_time;
            sum_sort += s.els_sort_time;
            sum_filter += s.els_filter_time;
            sum_total += s.els_total_time;

            sum_mask_gen += query_stats[repeat][LsearchId][i].mask_gen_time_ms;
            sum_global_sort += query_stats[repeat][LsearchId][i].global_sort_time_ms;
         }
         float avg_recall_for_batch = (num_queries > 0) ? (static_cast<float>(total_recall_for_batch) / num_queries) : 0.0f;
         int efs_for_batch = (num_queries > 0) ? (total_efs_for_batch / num_queries) : 0;
         double avg_ndc_for_batch = (num_queries > 0) ? (total_ndc_for_batch / num_queries) : 0.0;
         // Compute average time for each ELS stage
         double avg_trie = (num_queries > 0) ? sum_trie / num_queries : 0.0;
         double avg_sort = (num_queries > 0) ? sum_sort / num_queries : 0.0;
         double avg_filter = (num_queries > 0) ? sum_filter / num_queries : 0.0;
         double avg_total = (num_queries > 0) ? sum_total / num_queries : 0.0;

         double avg_mask_gen = (num_queries > 0) ? sum_mask_gen / num_queries : 0.0;
         double avg_global_sort = (num_queries > 0) ? sum_global_sort / num_queries : 0.0;

         // Store batch timing and average recall in the corresponding data structures
         // a. Store detailed_times for search_time_details.csv
         detailed_times.push_back({repeat, current_Lsearch, efs_for_batch, time_cost, avg_recall_for_batch, avg_trie, avg_sort, avg_filter, avg_total, avg_mask_gen, avg_global_sort});

         // b. Group by Lsearch for later summary averaging in search_time_summary.csv
         time_per_lsearch[current_Lsearch].push_back(time_cost);
         recall_per_lsearch[current_Lsearch].push_back(avg_recall_for_batch);
         efs_per_lsearch[current_Lsearch].push_back(efs_for_batch);

         std::cout << "  Lsearch=" << current_Lsearch << ", efs=" << efs_per_lsearch[current_Lsearch][0] 
                   << ", global pred time=" << total_global_pred_time << "ms"
                   << ", global sort time=" << total_global_sort_time << "ms"
                   << ", pure search time=" << pure_search_time << "ms"
                   << ", time=" << time_cost << "ms" << ", avg_recall=" << avg_recall_for_batch << std::endl;

      }
   }

   // save search_time_details.csv
   std::string details_file_path = result_path_prefix + "search_time_details.csv";
   std::ofstream details_out(details_file_path);
   if (details_out.is_open())
   {
      details_out << "Repeat,Lsearch,efs,Time_ms,Avg_Recall,Avg_Trie,Avg_Sort,Avg_Filter,Avg_Total,Avg_MaskGen,Avg_GlobalSort\n";
      for (const auto &log : detailed_times)
      {
         details_out << log.repeat << "," << log.l_search <<","<< log.efs << "," << log.time_ms << "," << log.avg_recall << "," 
                     << log.els_trie_avg << "," << log.els_sort_avg << "," << log.els_filter_avg << "," << log.els_total_avg << ","
                     << log.mask_gen_avg << "," << log.global_sort_avg << "\n";
      }
      details_out.close();
      std::cout << "\nDetailed search timing has been saved to: " << details_file_path << std::endl;
   }
   else
   {
      std::cerr << "Error: Unable to open file for writing: " << details_file_path << std::endl;
   }

   // save search_time_summary.csv
   std::string summary_file_path = result_path_prefix + "search_time_summary.csv";
   std::ofstream summary_out(summary_file_path);
   if (summary_out.is_open())
   {
      // 1. Update the header by adding the Average_Efs column
      summary_out << "Lsearch,Average_Efs,Average_Time_ms,Average_Recall\n";
      for (auto const &[l_search, times] : time_per_lsearch)
      {
         if (!times.empty())
         {
            // Compute average time
            double sum_time = std::accumulate(times.begin(), times.end(), 0.0);
            double avg_time = sum_time / times.size();

            // Compute average recall
            const auto &recalls = recall_per_lsearch.at(l_search);
            double sum_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0f);
            double avg_recall = sum_recall / recalls.size();

            // 2. Compute average efs
            const auto &efs_values = efs_per_lsearch.at(l_search);
            double sum_efs = std::accumulate(efs_values.begin(), efs_values.end(), 0.0);
            double avg_efs = sum_efs / efs_values.size();

            // 3. Write avg_efs to the file
            summary_out << l_search << "," << avg_efs << "," << avg_time << "," << avg_recall << "\n";
         }
      }
      summary_out.close();
      std::cout << "Performance summary (average efs/time/recall) has been saved to: " << summary_file_path << std::endl;
   }
   else
   {
      std::cerr << "Error: Unable to open file for writing: " << summary_file_path << std::endl;
   }

   // save query details
   std::ofstream detail_out(result_path_prefix + "query_details_repeat" + std::to_string(num_repeats) + ".csv");
   detail_out << "repeat,Lsearch,efs,QueryID,Time_ms,search_time_ms,core_search_time_ms,Recall,"         
              << "Algo_Choice,IsIntelElsUsed,IsTrieRec,"                                                         
              << "DistCalcs,NumNodeVisited,"                                                             
              << "MinSupersetT_ms,"
              << "ELS_TrieT_ms,ELS_SortT_ms,ELS_FilterT_ms,ELS_TotalT_ms,"
            //   << "IntelELS_PredT_ms,Route_PredT_ms,FpassT_ms,Routing_TotalT_ms,BitmapT_new_ms,FeatureT_ms," 
              << "IntelELS_PredT_ms,Route_PredT_ms,Mask_GenT_ms,Global_SortT_ms,FpassT_ms,Routing_TotalT_ms,BitmapT_new_ms,FeatureT_ms,"
              << "RabitQ_CtxPrepareT_ms,RabitQ_CtxRotateT_ms,RabitQ_CtxQ2CentroidsT_ms,RabitQ_CtxWrapperT_ms,"
              << "RabitQ_BinT_ms,RabitQ_FullT_ms,RabitQ_BinCalls,RabitQ_FullCalls,RabitQ_CtxReused,"
              << "AcornFilterType,"
              << "QuerySize,CandSize,ExactCandSize,GlobalPpass,"
              << "NumEntries,NumDescendants"
              << "\n";
   for (int repeat = 0; repeat < num_repeats; repeat++)
   {
      for (int LsearchId = 0; LsearchId < Lsearch_list.size(); LsearchId++)
      {
         for (int i = 0; i < num_queries; ++i)
         {
            const auto &stats = query_stats[repeat][LsearchId][i];
            detail_out << repeat << ","
                       << Lsearch_list[LsearchId] << ","
                       << stats.acorn_efs_used << ","
                       << i << ","
                       << stats.time_ms << ","
                       << stats.search_time_ms << ","
                       << stats.core_search_time_ms << ","
                       << stats.recall << ","
                       << stats.algo_choice << ","
                       << stats.is_intel_els_used << "," // <-- Whether the IntelELS model was actually invoked
                       << stats.is_trie_recursive << "," // <-- Whether nTtrue(1) or nTfalse(0) was used
                       << stats.num_distance_calcs << ","
                       << stats.num_nodes_visited << ","
                       << stats.get_min_super_sets_time_ms << ","
                       << stats.els_trie_time << ","
                        << stats.els_sort_time << ","
                        << stats.els_filter_time << ","
                        << stats.els_total_time << ","
                       << stats.intel_els_pred_time_ms << ","
                       << stats.route_pred_time_ms << ","
                       << stats.mask_gen_time_ms << ","       
                       << stats.global_sort_time_ms << ","    
                       << stats.fpass_time_ms << ","
                       << stats.routing_total_time_ms << ","
                       << stats.bitmap_time_ms << ","
                       << stats.feature_extract_time_ms << ","
                       << stats.rabitq_ctx_prepare_time_ms << ","
                       << stats.rabitq_ctx_rotate_time_ms << ","
                       << stats.rabitq_ctx_q2c_time_ms << ","
                       << stats.rabitq_ctx_wrapper_time_ms << ","
                       << stats.rabitq_bin_time_ms << ","
                       << stats.rabitq_full_time_ms << ","
                       << stats.rabitq_bin_calls << ","
                       << stats.rabitq_full_calls << ","
                       << stats.rabitq_ctx_reused << ","
                       << stats.acorn_filter_type << ","
                       // Idea1 and Trie features
                       << stats.query_length << ","
                       << stats.candidate_set_size << ","
                       << stats.exact_cand_size << ","  
                       << stats.global_p_pass << ","
                       << stats.num_entry_points << ","
                       << stats.num_lng_descendants <<"\n";
         }
      }
   }
   detail_out.close();
   
   if (navix_index) {
       delete navix_index;
   }

   std::cout << "- all done" << std::endl;
   return 0;
}
