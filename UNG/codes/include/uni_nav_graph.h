#ifndef UNG_H
#define UNG_H
#include "trie.h"
#include "graph.h"
#include "storage.h"
#include "distance.h"
#include "search_cache.h"
#include "label_nav_graph.h"
#include "rabitq_side_index.h"
#include "vamana/vamana.h"
#include "MethodSelector.h"
#include "ThreadPool.h"
#include "../../../ACORN/faiss/IndexACORN.h"
#include "../../../ACORN/faiss/index_io.h"
#include <unordered_map>
#include <bitset>
#include <optional>
#include <boost/dynamic_bitset.hpp>
#include <roaring/roaring.h>
#include <roaring/roaring.hh>
#include <faiss_navix/IndexHNSW.h>
#include <memory>
#ifdef ENABLE_KNOWHERE_MILVUS_BASELINE
#include <knowhere/index/index.h>
#include <knowhere/index/index_node.h>
#endif

using BitsetType = boost::dynamic_bitset<>;

namespace ANNS
{
   struct QueryStats
   {
      float recall;
      double time_ms;
      double search_time_ms;
      double core_search_time_ms;
      double descendants_merge_time_ms;  // Time spent merging descendants
      double coverage_merge_time_ms;     // Time spent merging coverage sets
      double get_min_super_sets_time_ms; // Time spent retrieving minimum entry sets
      double fpass_time_ms = 0.0;

      size_t num_distance_calcs;
      int acorn_efs_used;

      size_t num_nodes_visited = 0; // Total number of nodes visited during search
      size_t query_length;            // Query length
      long long trie_nodes_traversed; // Total Trie nodes traversed by both methods

      // Static Trie features
      size_t trie_total_nodes;
      size_t trie_label_cardinality;
      float trie_avg_path_length;
      float trie_avg_branching_factor;

      // ========= Idea 1 metrics =========
      // Method 1 related
      size_t candidate_set_size;
      size_t successful_checks = 0;
      float shortcut_hit_ratio = 0.0f;
      long long redundant_upward_steps = 0; // Duplicate nodes encountered during upward backtracking

      // Method 2 related
      size_t recursive_calls = 0;
      size_t pruning_events = 0;
      float pruning_efficiency = 0.0f;

      // ========= Idea 2 metrics =========
      size_t num_entry_points;
      size_t num_lng_descendants;
      float entry_group_total_coverage;


      size_t exact_cand_size = 0;
      float global_p_pass = 0.0f;

    float algo_choice;                   // Final selected algorithm ID, in the 0-7 range
    bool is_intel_els_used;             // True only when _trie_method_selector inference is actually invoked
    bool is_trie_recursive;               // Whether ELS used recursive nTtrue or non-recursive nTfalse

    double intel_els_pred_time_ms;       // Time spent predicting nTtrue/nTfalse
    double route_pred_time_ms;           // Time spent on SmartRoute/FastSmartRoute inference
    double l1_pred_time_ms = 0.0;
    double l2_pred_time_ms = 0.0;
    double routing_total_time_ms;        // Total routing-decision time, including features, prediction, and ELS computation
    double bitmap_time_ms;         
    double feature_extract_time_ms;

    int acorn_filter_type = 0; // ACORN mask type: 0=N/A, 1=ELS, 2=ExactMask, 3=InvertedIndex

    // ===== Detailed ELS timing =====
    double els_trie_time = 0;
    double els_sort_time = 0;
    double els_filter_time = 0;
    double els_total_time = 0;

      double global_sort_time_ms = 0.0;  // Amortized sorting time
      double mask_gen_time_ms = 0.0;     // Physical mask generation time

      // ===== Fine-grained RabitQ timing =====
      double rabitq_ctx_prepare_time_ms = 0.0;  // Query preprocessing time: rotate + q_to_centroids + wrapper
      double rabitq_ctx_rotate_time_ms = 0.0;   // Query rotation time
      double rabitq_ctx_q2c_time_ms = 0.0;      // Query-to-centroid distance precomputation time
      double rabitq_ctx_wrapper_time_ms = 0.0;  // Wrapper construction time
      double rabitq_bin_time_ms = 0.0;          // Cumulative bin-estimation time
      double rabitq_full_time_ms = 0.0;         // Cumulative full-refinement time
      size_t rabitq_bin_calls = 0;              // Number of bin calls
      size_t rabitq_full_calls = 0;             // Number of full-refinement calls
      bool rabitq_ctx_reused = false;           // Whether an existing context was reused for this query

      
   };
   struct NewEdgeCandidate
   {
      IdxType from;
      IdxType to;
      float distance;

      bool operator<(const NewEdgeCandidate &other) const
      {
         return distance < other.distance;
      }
   };
   struct AcornInUng
   {
      bool ung_and_acorn;
      std::string new_edge_policy;   // Run ACORN only, safeguard edges only, or both
      int R_in_add_new_edge;         // Number of neighbors ACORN retrieves per query
      int W_in_add_new_edge;         // Number of nearest neighbors retained from ACORN candidates
      int M_in_add_new_edge;         // Maximum number of new outgoing/incoming edges per point
      float layer_depth_retio;       // Ratio of candidate count to total vector count
      float query_vector_ratio;      // Ratio of query vectors to total candidate count
      float root_coverage_threshold; // Minimum coverage ratio for treating an attribute as a concept root
      std::string acorn_in_ung_output_path;

      // ACORN parameters
      int M, M_beta, gamma, efs, compute_recall;
   };

   // Entry-point selection strategy
   enum class SelectionMode
   {
      SizeOnly,       // Strategy 1: consider only group size
      SizeAndDistance // Strategy 2: combine group size and LNG hop distance
   };

   class UniNavGraph
   {
   public:
      enum class UngDistanceMode
      {
         Exact,
         RabitQ
      };

      UniNavGraph(IdxType num_nodes) : _label_nav_graph(std::make_shared<LabelNavGraph>(num_nodes)) {} // Initialize _label_nav_graph in the constructor
      UniNavGraph() = default;
      ~UniNavGraph() = default;

      void build(std::shared_ptr<IStorage> base_storage, std::shared_ptr<DistanceHandler> distance_handler,
                 std::string scenario, std::string index_name, uint32_t num_threads, IdxType num_cross_edges,
                 IdxType max_degree, IdxType Lbuild, float alpha, std::string dataset,
                 ANNS::AcornInUng new_cross_edge);

      void calculate_query_features_only(
          std::shared_ptr<IStorage> query_storage,
          uint32_t num_threads,
          const std::string &output_csv_path,
          bool is_new_trie_method,
          bool is_rec_more_start);
      void search(std::shared_ptr<IStorage> query_storage, std::shared_ptr<DistanceHandler> distance_handler,
                 uint32_t num_threads, IdxType Lsearch, IdxType num_entry_points, std::string scenario,
                 IdxType K, std::pair<IdxType, float> *results, std::vector<float> &num_cmps,
                 std::vector<std::bitset<10000001>> &bitmap);
      void search_hybrid(std::shared_ptr<IStorage> &query_storage,
                         std::shared_ptr<DistanceHandler> &distance_handler,
                         uint32_t num_threads, IdxType Lsearch,
                         IdxType num_entry_points, std::string scenario,
                         IdxType K, std::pair<IdxType, float> *results,
                         std::vector<float> &num_cmps,
                         std::vector<QueryStats> &query_stats,
                         bool is_new_trie_method, bool is_rec_more_start,
                         bool is_ung_more_entry,
                         int lsearch_start, int lsearch_step,
                         int efs_start, int efs_step_slow,int efs_step_fast,int lsearch_threshold, 
                         int routing_mode, int baseline_alg, faiss_navix::IndexHNSWFlat* navix_index = nullptr,
                         const std::vector<IdxType> &true_query_group_ids = {},// True source-group ID for each query
                         const std::vector<int>& query_algo_choices = {},
                         std::queue<int> task_queue = std::queue<int>(),bool optimize_standalone_prefilter = false); 

      // Global prediction: compute masks and model routing for all queries
      std::vector<int> global_predict_algo_choices(
          std::shared_ptr<IStorage> query_storage,
          int routing_mode,
          const std::vector<int>& csv_choices,
          uint32_t num_threads,
          std::vector<QueryStats>& out_global_stats);

      // Global ordering: cache-friendly reordering by algorithm and hash
      std::vector<int> get_sorted_query_ids(
          std::shared_ptr<IStorage> query_storage,
          const std::vector<int>& algo_choices,
          int routing_mode);

      // I/O
      void save(std::string index_path_prefix, std::string results_path_prefix);
      void load(std::string index_path_prefix, std::string selector_modle_prefix, const std::string &data_type, const std::string &acorn_index_path, const std::string &acorn_1_index_path, const std::string &dataset);
      void configure_rabitq_build(bool enable, size_t total_bits);
      void set_ung_distance_mode(const std::string &mode);
      void prepare_rabitq_query_contexts(std::shared_ptr<IStorage> &query_storage,
                                         const std::string &query_bin_file = "");

      // query generator

      std::map<std::vector<unsigned int>, int> _subset_count_cache; // Cache label-combination frequencies to avoid repeated expensive computation. The key is the sorted label subset, and the value is its dataset-wide frequency.
      int count_subset_occurrences(const std::vector<unsigned int> &sorted_subset);
      void query_generate(std::string &output_prefix, int n, float keep_prob, int K, bool stratified_sampling, bool verify);
      void generate_multiple_queries(std::string dataset,
                                     UniNavGraph &index,
                                     int K,
                                     const std::string &base_output_path,
                                     int num_sets,
                                     int n_per_set,
                                     float keep_prob,
                                     bool stratified_sampling,
                                     bool verify);
      void generate_queries_method1_high_coverage(std::string &output_prefix, std::string dataset, int query_n, std::string &base_label_file, float coverage_threshold);
      void generate_queries_method1_low_coverage(
          std::string &output_prefix,
          std::string dataset,
          int query_n,
          std::string &base_label_file,
          int num_of_per_query_labels,
          float coverage_threshold,
          int K);
      void generate_queries_method2_high_coverage(int N, int K, int top_M_trees, std::string dataset, const std::string &output_prefix, const std::string &base_label_tree_roots);
      void generate_queries_method2_high_coverage_human(
          std::string &output_prefix,
          std::string dataset,
          int query_n,
          std::string &base_label_file,
          std::string &base_label_info_file);
      void generate_queries_method2_low_coverage(
          std::string &output_prefix,
          std::string dataset,
          int query_n,
          std::string &base_label_file,
          int num_of_per_query_labels,
          int K,
          int max_K,
          int min_K);
      void generate_queries_true_data_high_coverage(
          int N,                            // Total number of queries to generate
          int K,                            // Number of neighbors used to compute centroids
          int top_M_trees,                  // Select the M concept trees with the highest coverage
          std::string dataset,              // Dataset name used to generate filenames
          const std::string &output_prefix, // Output file path prefix
          float min_root_coverage_threshold);
      void generate_queries_true_data_low_coverage(
          std::string &output_prefix,
          std::string dataset,
          int query_n,
          std::string &base_label_file,
          int num_of_per_query_labels,
          float coverage_threshold,
          int K);
      void generate_queries_hard_sandwich(
          int N,                            // Total number of queries to generate
          const std::string &output_prefix, // Output file path prefix
          const std::string &dataset,       // Dataset name
          float parent_min_coverage_ratio,  // Minimum coverage threshold for parent nodes, for example 0.02
          float child_max_coverage_ratio,   // Maximum coverage threshold for child nodes, for example 0.005
          float query_min_selectivity,      // Minimum query selectivity, for example 0.0005 or 0.05%
          float query_max_selectivity);     // Maximum query selectivity, for example 0.01 or 1%
      std::vector<int> _lng_node_depths;    // Graph depth for each group_id, used by generate_queries_hard_sandwich
      void _precompute_lng_node_depths();
      void generate_queries_hard_top_n_rare(
          int N,                            // Total number of queries to generate
          const std::string &output_prefix, // Output file path prefix
          const std::string &dataset,       // Dataset name
          int num_rare_labels_to_use,       // Number of lowest-frequency labels to use
          float query_min_selectivity,      // Minimum query selectivity
          float query_max_selectivity,      // Maximum query selectivity
          int min_frequency_for_rare_labels = 1);

      void load_bipartite_graph(const std::string &filename);
      bool compare_graphs(const ANNS::UniNavGraph &g1, const ANNS::UniNavGraph &g2);
      IdxType _num_points;
      std::vector<std::vector<IdxType>> _vector_attr_graph; // Graph represented as adjacency lists
      std::unordered_map<LabelType, AtrType> _attr_to_id;   // Attribute-to-ID mapping
      std::unordered_map<AtrType, LabelType> _id_to_attr;   // ID-to-attribute mapping
      AtrType _num_attributes;                              // Number of unique attributes

      std::pair<std::bitset<10000001>, double> compute_attribute_bitmap(const std::vector<LabelType> &query_attributes) const; // Build bitmap
      roaring::Roaring compute_bitmap_from_groups(const std::vector<IdxType> &group_ids) const;
      std::vector<roaring::Roaring> batch_compute_ung_bitmaps(
          const ANNS::UniNavGraph &index,
          const std::shared_ptr<ANNS::IStorage> &query_storage,
          uint32_t num_threads,
          bool is_new_trie_method,
          bool is_rec_more_start);

    
    // Basic bitmap plus brute-force search
    void search_baseline_exact(
      const char* query,
      const std::bitset<16000000>& final_bitmap,
      IdxType K,
      std::pair<IdxType, float>* results,
      size_t& num_distance_calcs,
      bool use_optimized_bitset);
    const std::bitset<16000000>& get_exact_cand_size_and_mask(
      const std::vector<LabelType>& query_labels,
      size_t& cand_size,
      bool use_optimized = true) const;

    // Use CRoaring for bitmap computation
    std::vector<roaring::Roaring> _vec_attr_roaring_inv;// Vector-level CRoaring inverted index for fast GlobalPpass computation
    void build_vector_inverted_indices();
    void search_baseline_exact_roaring(
        const char* query,
        const roaring::Roaring& valid_bitmap,
        IdxType K,
        std::pair<IdxType, float>* results,
        size_t& num_distance_calcs);
    void search_baseline_rabitq(
      const std::bitset<16000000>& final_bitmap,
      IdxType K,
      std::pair<IdxType, float>* results,
      size_t& num_distance_calcs,
      ANNS::rabitq::RabitQSideIndex::QueryContext& query_ctx,
      QueryStats& stats,
      bool use_optimized_bitset,
      const ANNS::rabitq::RabitQSideIndex* side_index = nullptr);
    void search_baseline_rabitq_roaring(
      const roaring::Roaring& valid_bitmap,
      IdxType K,
      std::pair<IdxType, float>* results,
      size_t& num_distance_calcs,
      ANNS::rabitq::RabitQSideIndex::QueryContext& query_ctx,
      QueryStats& stats,
      const ANNS::rabitq::RabitQSideIndex* side_index = nullptr);
    bool run_milvus_knowhere_baseline(
      IdxType query_id,
      const char* query,
      const std::vector<LabelType>& query_labels,
      IdxType Lsearch,
      IdxType K,
      int milvus_baseline_alg,
      SearchQueue& cur_result,
      QueryStats& stats,
      float& num_cmps_out);


      // Data structures required for search flags
      std::vector<BitsetType> _lng_descendants_bits; // Descendant set for each group
      std::vector<BitsetType> _covered_sets_bits;    // Coverage set for each group
      std::vector<roaring::Roaring> _lng_descendants_rb;
      std::vector<roaring::Roaring> _covered_sets_rb;

      std::vector<IdxType> select_entry_groups(
          const std::vector<IdxType> &minimum_entry_sets, // Required base entry points
          SelectionMode mode,                             // Selection strategy
          size_t top_k,                                   // Number of additional best entry points to select
          double beta = 1.0,                              // Beta weight for hop distance in Strategy 2
          IdxType true_query_group_id = 0) const;         // Const because this function should not mutate graph state

      void get_min_super_sets_debug(const std::vector<LabelType> &query_label_set,
                                    std::vector<IdxType> &min_super_set_ids,
                                    bool avoid_self, bool need_containment,
                                    std::atomic<int> &print_counter, bool is_new_trie_method, bool is_rec_more_start, QueryStats &stats,
                                    bool skip_group_id_check);

      void warmup_selectors(uint32_t num_threads);// Warm up selector models to avoid first-query latency



    // Lightweight inverted adjacency list for Method 2: [AttrID] -> [GroupID1, GroupID2, ...]
    std::vector<std::vector<IdxType>> _group_attr_adj_list; 
    // CRoaring inverted index for Method 3: [AttrID] -> RoaringBitmap(GroupIDs)
    std::vector<roaring::Roaring> _group_attr_roaring_inv;
    void build_group_inverted_indices();// Build inverted indexes
    void evaluate_fpass_methods(std::shared_ptr<IStorage> query_storage, const std::string& output_csv_path);// Benchmark the five Fpass computation methods

    std::vector<int> load_query_algo_choices_from_csv(
    const std::string &csv_path,
    size_t expected_num_queries) const;

    // Dynamically control whether to skip the ELS filter
    bool skip_els_filter = false;

    // Reusable class-level thread pool
    std::unique_ptr<ThreadPool> _thread_pool = nullptr;

   private:

      void thread_function(int id, SearchCacheList& search_cache_list,
                           std::shared_ptr<IStorage> &query_storage,
                                   std::shared_ptr<DistanceHandler> &distance_handler,
                                   uint32_t num_threads, IdxType Lsearch,
                                   IdxType num_entry_points, std::string scenario,
                                   IdxType K, std::pair<IdxType, float> *results,
                                   std::vector<float> &num_cmps,
                                   std::vector<QueryStats> &query_stats,
                                   bool is_new_trie_method, bool is_rec_more_start,
                                   bool is_ung_more_entry,
                                   int lsearch_start, int lsearch_step,
                                   int efs_start, int efs_step_slow,int efs_step_fast,int lsearch_threshold,
                                   int routing_mode,int baseline_alg, IdxType num_queries, 
                                   faiss_navix::IndexHNSWFlat* navix_index,
                           const std::vector<IdxType> &true_query_group_ids,const std::vector<int> &query_algo_choices,
                           bool optimize_standalone_prefilter);
      size_t get_candidate_count_for_label(LabelType label) const;
      // data
      std::shared_ptr<IStorage> _base_storage,
          _query_storage;
      std::shared_ptr<DistanceHandler> _distance_handler;
      std::shared_ptr<Graph> _graph;
      // IdxType _num_points;

      // trie index and vector groups
      IdxType _num_groups;
      TrieIndex _trie_index;
      std::vector<IdxType> _new_vec_id_to_group_id;
      std::vector<std::vector<IdxType>> _group_id_to_vec_ids;
      std::vector<std::vector<LabelType>> _group_id_to_label_set;
      void build_trie_and_divide_groups();

      // label navigating graph
      std::shared_ptr<LabelNavGraph> _label_nav_graph = nullptr;
      void get_min_super_sets(const std::vector<LabelType> &query_label_set, std::vector<IdxType> &min_super_set_ids,
                              bool avoid_self = false, bool need_containment = true);
      void cal_f_coverage_ratio();
      void build_label_nav_graph();
      size_t count_all_descendants(IdxType group_id) const;
      void print_lng_descendants_num(const std::string &filename) const;
      void get_descendants_info();

      // prepare vector storage for each group
      std::vector<IdxType> _new_to_old_vec_ids;
      std::vector<IdxType> _old_to_new_vec_ids; // fxy_add
      std::vector<std::pair<IdxType, IdxType>> _group_id_to_range;
      std::vector<std::shared_ptr<IStorage>> _group_storages;
      void prepare_group_storages_graphs();

      // graph indices for each graph
      std::string _index_name;
      std::vector<std::shared_ptr<Graph>> _group_graphs;
      std::vector<IdxType> _group_entry_points;
      void build_graph_for_all_groups();
      void build_complete_graph(std::shared_ptr<Graph> graph, IdxType num_points);
      std::vector<std::shared_ptr<Vamana>> _vamana_instances;

      std::shared_ptr<Graph> _global_graph;
      std::shared_ptr<Vamana> _global_vamana; // Global Vamana instance
      IdxType _global_vamana_entry_point;     // Entry point of the global Vamana instance
      void build_global_vamana_graph();

      void build_vector_and_attr_graph();
      size_t count_graph_edges() const;
      void save_bipartite_graph_info() const;
      void save_bipartite_graph(const std::string &filename);
      uint32_t compute_checksum() const;
      // void load_bipartite_graph(const std::string &filename);

      // Flag-handling helpers
      void initialize_lng_descendants_coverage_bitsets();
      void initialize_roaring_bitsets();

      // Add new cross-group edges
      void add_new_distance_oriented_edges(
          const std::string &dataset,
          uint32_t num_threads,
          ANNS::AcornInUng new_cross_edge);
      int _num_distance_oriented_edges;
      const bool ENABLE_SEARCH_PATH_LOGGING = true;
      std::unordered_set<uint64_t> _my_new_edges_set;

      void finalize_intra_group_graphs(); // Convert local graph IDs to global IDs

      // index parameters for each graph
      IdxType _max_degree,
          _Lbuild;
      float _alpha;
      uint32_t _num_threads;
      std::string _scenario;

      // cross-group edges
      IdxType _num_cross_edges;
      std::vector<SearchQueue> _cross_group_neighbors;
      void build_cross_group_edges();

      // obtain the final unified navigating graph
      void add_offset_for_uni_nav_graph();

      // obtain entry_points
      std::vector<IdxType> get_entry_points(const std::vector<LabelType> &query_label_set,
                                            IdxType num_entry_points, VisitedSet &visited_set);
      void get_entry_points_given_group_id(IdxType num_entry_points, VisitedSet &visited_set,
                                           IdxType group_id, std::vector<IdxType> &entry_points);

      // search in graph
      IdxType iterate_to_fixed_point(const char *query, std::shared_ptr<SearchCache> search_cache,
                                     IdxType target_id, const std::vector<IdxType> &entry_points,
                                     size_t &num_nodes_visited,
                                     bool clear_search_queue = true, bool clear_visited_set = true);
      IdxType iterate_to_fixed_point_rabitq(const char *query, std::shared_ptr<SearchCache> search_cache,
                                            IdxType target_id, const std::vector<IdxType> &entry_points,
                                            size_t &num_nodes_visited,
                                            QueryStats &stats,
                                            ANNS::rabitq::RabitQSideIndex::QueryContext *cached_query_ctx = nullptr,
                                            bool clear_search_queue = true, bool clear_visited_set = true);
      // search in global graph
      IdxType iterate_to_fixed_point_global(const char *query, std::shared_ptr<SearchCache> search_cache,
                                            IdxType target_id, const std::vector<IdxType> &entry_points,
                                            bool clear_search_queue = true, bool clear_visited_set = true);

      // statistics
      float _index_time = 0, _label_processing_time = 0, _build_graph_time = 0, _build_vector_attr_graph_time = 0, _cal_descendants_time = 0, _cal_coverage_ratio_time = 0;
      float _build_LNG_time = 0, _build_cross_edges_time = 0;
      double _build_roaring_bitsets_time;
      float _index_size, _index_size_add_rb;
      IdxType _graph_num_edges, _LNG_num_edges;

      // FXY_ADD: detailed timing for add_new_distance_oriented_edges
      double _cross_edge_step1_time_ms;                     // Step 1 time: identification and sampling
      double _cross_edge_step2_acorn_time_ms;               // Step 2 time: ACORN execution
      double _cross_edge_step3_add_dist_edges_time_ms;      // Step 3 time: adding distance-oriented edges
      double _cross_edge_step4_add_hierarchy_edges_time_ms; // Step 4 time: adding hierarchy safeguard edges
      void statistics();

      std::string _dataset;
      bool _build_rabitq_side_index = false;
      size_t _rabitq_total_bits = 4;
      double _rabitq_build_time_ms = 0.0;
      uint64_t _rabitq_side_size_bytes = 0;
      UngDistanceMode _ung_distance_mode = UngDistanceMode::Exact;
      ANNS::rabitq::RabitQSideIndex _rabitq_side_index;
      ANNS::rabitq::RabitQSideIndex _acorn_rabitq_side_index;
      ANNS::rabitq::RabitQSideIndex _acorn_1_rabitq_side_index;
      const IStorage *_rabitq_cached_query_storage = nullptr;
      std::vector<std::unique_ptr<ANNS::rabitq::RabitQSideIndex::QueryContext>> _rabitq_query_ctx_cache;
      std::vector<double> _rabitq_query_ctx_prepare_ms;
      std::vector<double> _rabitq_query_ctx_rotate_ms;
      std::vector<double> _rabitq_query_ctx_q2c_ms;
      std::vector<double> _rabitq_query_ctx_wrapper_ms;

      // idea1 selector
      std::unique_ptr<MethodSelector> _trie_method_selector;
      TrieStaticMetrics _trie_static_metrics; // Cache static Trie metrics to avoid repeated computation

      // idea2 selector
      std::shared_ptr<faiss::IndexACORNFlat> _acorn_index;
      std::shared_ptr<faiss::IndexACORNFlat> _acorn_1_index;
#ifdef ENABLE_KNOWHERE_MILVUS_BASELINE
      knowhere::Index<knowhere::IndexNode> _milvus_knowhere_index;
      bool _milvus_knowhere_ready = false;
      bool _milvus_knowhere_is_ivf = true;
      int _milvus_knowhere_nlist = 4096;
      int _milvus_knowhere_nprobe = 16;
      knowhere::Index<knowhere::IndexNode> _milvus_knowhere_hnsw_index;
      bool _milvus_knowhere_hnsw_ready = false;
      int _milvus_knowhere_hnsw_m = 32;
      int _milvus_knowhere_hnsw_efc = 200;
      int _milvus_knowhere_hnsw_ef = 100;
#endif
      std::unique_ptr<MethodSelector> _ung_acorn_selector;
      std::optional<bool> check_idea2_heuristic_override(const std::string& dataset_name, size_t num_entry_groups) const;
      std::optional<bool> check_pre_trie_heuristic(const std::string& dataset_name, size_t query_length, size_t candidate_set_size) const;

      // smartroute selector
      std::unique_ptr<MethodSelector> _smart_route_selector;    
      int _majority_acorn_id = 2; // Default to ACORN-gamma (id: 2)
      std::unique_ptr<MethodSelector> _fast_route_single_selector;
      int _single_majority_acorn_id = 2;
      std::unique_ptr<MethodSelector> _fast_route_revised_selector;
      int _revised_majority_acorn_id = 2;
      int determine_routing_strategy(
        int routing_mode, 
        int baseline_alg,
        const std::vector<LabelType>& query_labels,
        ANNS::QueryStats& stats,
        std::vector<IdxType>& entry_group_ids,
        bool is_new_trie_method,
        bool is_rec_more_start);
   };
}

#endif // UNG_H
