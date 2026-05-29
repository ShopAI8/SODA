#include <chrono>
#include <fstream>
#include <iostream>
#include <boost/program_options.hpp>
#include "uni_nav_graph.h"
#include <omp.h>

namespace po = boost::program_options;

int main(int argc, char **argv)
{

   // common auguments
   std::string data_type, dist_fn, base_bin_file, base_label_file, base_label_info_file, base_label_tree_roots, index_path_prefix, result_path_prefix;
   uint32_t num_threads;
   ANNS::IdxType num_cross_edges;

   // parameters for graph indices
   std::string index_type, scenario;
   ANNS::IdxType max_degree, Lbuild; // Vamana
   float alpha;                      // Vamana
   bool build_rabitq_side_index = false;
   size_t rabitq_total_bits = 4;

   // parameters for acorn in ung (hardcoded to default values)
   bool ung_and_acorn = false;
   std::string new_edge_policy = "false";
   int R_in_add_new_edge = 50, W_in_add_new_edge = 50, M_in_add_new_edge = 4;
   float layer_depth_retio = 0.8, query_vector_ratio = 0.8, root_coverage_threshold = 0.4;
   std::string acorn_in_ung_output_path = "false";
   int M = 32, M_beta = 64, gamma = 80, efs = 1000, compute_recall = 1;

   // if query file is not provided, generate query file (hardcoded to default values)
   bool generate_query = false;
   std::string generate_query_task = "false";
   std::string query_file_path = "my_words_query";
   std::string dataset;
   float method1_high_coverage_p = 0.7f;

   try
   {
      po::options_description desc{"Arguments"};

      // common arguments
      desc.add_options()("help,h", "Print information on arguments");
      desc.add_options()("dataset", po::value<std::string>(&dataset)->required(),
                         "dataset");
      desc.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                         "data type <int8/uint8/float>");
      desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                         "distance function <L2/IP/cosine>");
      desc.add_options()("base_bin_file", po::value<std::string>(&base_bin_file)->required(),
                         "File containing the base vectors in binary format");
      desc.add_options()("base_label_file", po::value<std::string>(&base_label_file)->required(),
                         "Base label file in txt format");
      desc.add_options()("base_label_info_file", po::value<std::string>(&base_label_info_file)->required(),
                         "Base label info file in log format");
      desc.add_options()("base_label_tree_roots", po::value<std::string>(&base_label_tree_roots)->required(),
                         "base_label_tree_roots");
      desc.add_options()("num_threads", po::value<uint32_t>(&num_threads)->default_value(1),
                         "Number of threads to use");
      desc.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                         "Path prefix for saving the index");
      desc.add_options()("result_path_prefix", po::value<std::string>(&result_path_prefix)->required(),
                         "Path prefix for saving the results");

      // parameters for graph indices
      desc.add_options()("scenario", po::value<std::string>(&scenario)->default_value("general"),
                         "Scenario for building UniNavGraph, <equality/general>");
      desc.add_options()("index_type", po::value<std::string>(&index_type)->default_value("Vamana"),
                         "Type of index to build, <Vamana>");
      desc.add_options()("num_cross_edges", po::value<ANNS::IdxType>(&num_cross_edges)->default_value(ANNS::default_paras::NUM_CROSS_EDGES),
                         "Number of cross edges for building Vamana");
      desc.add_options()("max_degree", po::value<ANNS::IdxType>(&max_degree)->default_value(ANNS::default_paras::MAX_DEGREE),
                         "Max degree for building Vamana");
      desc.add_options()("Lbuild", po::value<uint32_t>(&Lbuild)->default_value(ANNS::default_paras::L_BUILD),
                         "Size of candidate set for building Vamana");
      desc.add_options()("alpha", po::value<float>(&alpha)->default_value(ANNS::default_paras::ALPHA),
                         "Alpha for building Vamana");
      desc.add_options()("build_rabitq_side_index", po::value<bool>(&build_rabitq_side_index)->default_value(false),
                         "Build and persist RabitQ side index for UNG query acceleration (disabled unless UNG_ENABLE_RABITQ=ON)");
      desc.add_options()("rabitq_total_bits", po::value<size_t>(&rabitq_total_bits)->default_value(4),
                         "Total quantization bits for RabitQ side index [1..9]");

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
   if (scenario != "general" && scenario != "equality")
   {
      std::cerr << "Invalid scenario: " << scenario << std::endl;
      return -1;
   }
   if (build_rabitq_side_index && (rabitq_total_bits < 1 || rabitq_total_bits > 9))
   {
      std::cerr << "Invalid rabitq_total_bits: " << rabitq_total_bits << ", expected in [1, 9]." << std::endl;
      return -1;
   }
#if !UNG_ENABLE_RABITQ
   if (build_rabitq_side_index)
   {
      std::cerr << "[RabitQ] RabitQ support is disabled at compile time. Ignoring --build_rabitq_side_index." << std::endl;
      build_rabitq_side_index = false;
   }
#endif

   // load base data
   std::shared_ptr<ANNS::IStorage> base_storage = ANNS::create_storage(data_type);
   base_storage->load_from_file(base_bin_file, base_label_file);

   // preparation
   std::cout << "Building Unified Navigating Graph index based on " << index_type << " algorithm ..." << std::endl;
   std::shared_ptr<ANNS::DistanceHandler> distance_handler = ANNS::get_distance_handler(data_type, dist_fn);

   // AcornInUng new_cross_edge
   ANNS::AcornInUng new_cross_edge;
   new_cross_edge.ung_and_acorn = ung_and_acorn;
   new_cross_edge.new_edge_policy = new_edge_policy;
   new_cross_edge.R_in_add_new_edge = R_in_add_new_edge;
   new_cross_edge.W_in_add_new_edge = W_in_add_new_edge;
   new_cross_edge.M_in_add_new_edge = M_in_add_new_edge;
   new_cross_edge.layer_depth_retio = layer_depth_retio;
   new_cross_edge.query_vector_ratio = query_vector_ratio;
   new_cross_edge.root_coverage_threshold = root_coverage_threshold;
   new_cross_edge.M = M;
   new_cross_edge.M_beta = M_beta;
   new_cross_edge.gamma = gamma;
   new_cross_edge.efs = efs;
   new_cross_edge.compute_recall = compute_recall;
   new_cross_edge.acorn_in_ung_output_path = acorn_in_ung_output_path;

   // build index
   ANNS::UniNavGraph index;
   index.configure_rabitq_build(build_rabitq_side_index, rabitq_total_bits);
   auto start_time = std::chrono::high_resolution_clock::now();
   std::cout << "new_cross_edge.ung_and_acorn: " << new_cross_edge.ung_and_acorn << std::endl;
   index.build(base_storage, distance_handler, scenario, index_type, num_threads, num_cross_edges, max_degree, Lbuild, alpha, dataset, new_cross_edge);
   std::cout << "Index time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << "ms" << std::endl;

   // save index
   index.save(index_path_prefix, result_path_prefix);

   return 0;
}
