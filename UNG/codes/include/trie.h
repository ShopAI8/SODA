#ifndef TRIE_TREE_H
#define TRIE_TREE_H

#include <vector>
#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include "config.h"

namespace ANNS
{

   // fxy_add: Metrics passed by Method 1 (Shortcut)
   struct TrieMethod1Metrics
   {
      size_t initial_candidates = 0;
      size_t successful_checks = 0;

      long long upward_traversals;   // Number of nodes visited during upward backtracking
      long long bfs_nodes_processed; // Number of nodes processed during downward BFS

      long long redundant_upward_steps = 0; // Duplicate nodes encountered during upward backtracking

      // Timing fields for the two phases
      double time_phase1_ms = 0.0;
      double time_phase2_ms = 0.0;
   };

   // fxy_add: Collect detailed performance metrics for Method 2 (recursive search)
   struct TrieSearchMetricsRecursive
   {
      // --- Recursive search phase (DFS) ---
      long long recursive_calls = 0; // Total number of recursive calls
      long long pruning_events = 0;  // Number of pruning events
      int max_recursion_depth = 0;   // Maximum recursion depth reached

      // --- Result collection phase (BFS) ---
      long long collection_calls = 0;       // Number of collect_all_terminals calls
      long long nodes_processed_in_bfs = 0; // Total nodes processed by BFS across all collections
      double time_in_collection_bfs = 0.0;  // Total BFS time across all collections
   };

   // fxy_add:Add a struct to hold the calculated metrics
   struct TrieStaticMetrics
   {
      size_t label_cardinality = 0;
      size_t total_nodes = 0;
      float avg_path_length = 0.0;
      float avg_branching_factor = 0.0;
      std::map<ANNS::LabelType, size_t> label_frequency;
   };

   // trie tree node
   struct TrieNode
   {
      LabelType label;
      IdxType group_id;         // group_id>0, and 0 if not a terminal node
      LabelType label_set_size; // number of elements in the label set if it is a terminal node
      IdxType group_size;       // number of elements in the group if it is a terminal node

      std::shared_ptr<TrieNode> parent;
      // std::map<LabelType, std::shared_ptr<TrieNode>> children;
      std::unordered_map<LabelType, std::shared_ptr<TrieNode>> children;

      TrieNode(LabelType x, std::shared_ptr<TrieNode> y)
          : label(x), parent(y), group_id(0), label_set_size(0), group_size(0) {}
      TrieNode(LabelType a, IdxType b, LabelType c, IdxType d)
          : label(a), group_id(b), label_set_size(c), group_size(d) {}
      ~TrieNode() = default;
   };

   // trie tree construction and search for super sets
   class TrieIndex
   {

   public:
      TrieIndex();

      // construction
      IdxType insert(const std::vector<LabelType> &label_set, IdxType &new_label_set_id);

      // query
      LabelType get_max_label_id() const { return _max_label_id; }
      std::shared_ptr<TrieNode> find_exact_match(const std::vector<LabelType> &label_set) const;
      void get_super_set_entrances(const std::vector<LabelType> &label_set,
                                   std::vector<std::shared_ptr<TrieNode>> &super_set_entrances,
                                   bool avoid_self = false, bool need_containment = true) const;
      void get_super_set_entrances_debug(const std::vector<LabelType> &label_set,
                                         std::vector<std::shared_ptr<TrieNode>> &super_set_entrances,
                                         bool avoid_self, bool need_containment,
                                         std::atomic<int> &print_counter, TrieMethod1Metrics &metrics,
                                         bool skip_group_id_check) const;
      void get_super_set_entrances_new_debug(const std::vector<LabelType> &label_set,
                                             std::vector<std::shared_ptr<TrieNode>> &super_set_entrances,
                                             bool avoid_self, bool need_containment,
                                             std::atomic<int> &print_counter, TrieSearchMetricsRecursive &metrics) const;
      void get_super_set_entrances_new_more_sp_debug(const std::vector<LabelType> &label_set,
                                                     std::vector<std::shared_ptr<TrieNode>> &super_set_entrances,
                                                     bool avoid_self, bool need_containment,
                                                     std::atomic<int> &print_counter, TrieSearchMetricsRecursive &metrics) const;

      // fxy_add
      size_t get_candidate_count_for_label(LabelType label) const
      {
         // Step 1: Check whether the label ID is within the valid _label_to_nodes range
         if (label >= _label_to_nodes.size())
         {
            return 0; // Out-of-range labels cannot have candidate sets
         }
         // Step 2: Access by index directly and return the internal vector size
         return _label_to_nodes[label].size();
      }
      // fxy_add: Compute and return static structural metrics for the Trie
      TrieStaticMetrics calculate_static_metrics() const;

      // I/O
      void save(std::string filename) const;
      void load(std::string filename);
      float get_index_size();

      std::vector<std::vector<std::shared_ptr<TrieNode>>> _label_to_nodes;

   private:
      LabelType _max_label_id = 0;
      std::shared_ptr<TrieNode> _root;
      
      // help function for get_super_set_entrances
      bool examine_smallest(const std::vector<LabelType> &label_set, const std::shared_ptr<TrieNode> &node) const;
      bool examine_containment(const std::vector<LabelType> &label_set, const std::shared_ptr<TrieNode> &node) const;
      // bool examine_containment_debug(const std::vector<LabelType> &label_set, const std::shared_ptr<TrieNode> &node, long long &nodes_traversed) const;
      bool examine_containment_debug(const std::vector<LabelType> &label_set,
                                     const std::shared_ptr<TrieNode> &node,
                                     long long &nodes_traversed,
                                     std::unordered_set<std::shared_ptr<TrieNode>> &visited_upward,
                                     long long &redundant_steps) const;
      void find_supersets_recursive_debug(
          std::shared_ptr<TrieNode> current_node,
          const std::vector<LabelType> &sorted_query,
          size_t query_idx,
          std::vector<std::shared_ptr<TrieNode>> &results,
          std::set<IdxType> &visited_groups,
          const std::shared_ptr<TrieNode> &avoided_node,
          TrieSearchMetricsRecursive &metrics,
          int current_depth) const;

      void find_supersets_iterative_debug( // Iterative version
          std::shared_ptr<TrieNode> start_node,
          const std::vector<LabelType> &sorted_query,
          size_t start_query_idx,
          std::vector<std::shared_ptr<TrieNode>> &results,
          std::set<IdxType> &visited_groups,
          const std::shared_ptr<TrieNode> &avoided_node,
          TrieSearchMetricsRecursive &metrics) const;

      void collect_all_terminals_debug(
          std::shared_ptr<TrieNode> start_node,
          std::vector<std::shared_ptr<TrieNode>> &results,
          std::set<IdxType> &visited_groups,
          const std::shared_ptr<TrieNode> &avoided_node,
          TrieSearchMetricsRecursive &metrics) const;
   };
}

#endif // TRIE_TREE_H
