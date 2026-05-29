#include <set>
#include "utils.h"

namespace ANNS
{

   void write_kv_file(const std::string &filename, const std::map<std::string, std::string> &kv_map)
   {
      std::ofstream out(filename);
      for (auto &kv : kv_map)
      {
         out << kv.first << "=" << kv.second << std::endl;
      }
      out.close();
   }

   std::map<std::string, std::string> parse_kv_file(const std::string &filename)
   {
      std::map<std::string, std::string> kv_map;
      std::ifstream in(filename);
      std::string line;
      while (std::getline(in, line))
      {
         size_t pos = line.find("=");
         if (pos == std::string::npos)
            continue;
         std::string key = line.substr(0, pos);
         std::string value = line.substr(pos + 1);
         kv_map[key] = value;
      }
      in.close();
      return kv_map;
   }

   void write_gt_file(const std::string &filename, const std::pair<IdxType, float> *gt, uint32_t num_queries, uint32_t K)
   {
      std::ofstream fout(filename, std::ios::binary);
      fout.write(reinterpret_cast<const char *>(gt), num_queries * K * sizeof(std::pair<IdxType, float>));
      std::cout << "Ground truth written to " << filename << std::endl;
   }

   void load_gt_file(const std::string &filename, std::pair<IdxType, float> *gt, uint32_t num_queries, uint32_t K)
   {
      std::ifstream fin(filename, std::ios::binary);
      fin.read(reinterpret_cast<char *>(gt), num_queries * K * sizeof(std::pair<IdxType, float>));
      std::cout << "Ground truth loaded from " << filename << std::endl;
   }

   // fxy_add
   float calculate_recall(const std::pair<IdxType, float> *gt, const std::pair<IdxType, float> *results, uint32_t num_queries, uint32_t K)
   {
      float total_correct = 0;
      float total_relevant = 0; // Total number of true relevant results across all queries

      for (uint32_t i = 0; i < num_queries; i++)
      {
         // Build the ground-truth set and count true relevant results for the current query
         std::set<IdxType> gt_set;
         int32_t offset = -1;
         uint32_t num_relevant = 0; // Number of true relevant results for the current query

         for (uint32_t j = 0; j < K; j++)
         {
            if (gt[i * K + j].first != -1)
            {
               offset = j;
               gt_set.insert(gt[i * K + j].first);
               num_relevant++; // Count valid ground-truth entries
            }
         }

         total_relevant += num_relevant; // Accumulate into the global relevant-result count

         // Count correct matches
         for (uint32_t j = 0; j < K; j++)
         {
            if (results[i * K + j].first == -1)
               break;

            if (offset >= 0 && results[i * K + j].second == gt[i * K + offset].second)
            { // Tie case
               total_correct++;
               offset--;
            }
            else
            {
               if (gt_set.find(results[i * K + j].first) != gt_set.end())
                  total_correct++;
            }
         }
      }

      // Recall = correct matches / total true relevant results
      return (total_relevant > 0) ? (100.0f * total_correct / total_relevant) : 0.0f;
   }

   // fxy_add
   float calculate_recall_to_csv(const std::pair<IdxType, float> *gt,
                                 const std::pair<IdxType, float> *results,
                                 uint32_t num_queries,
                                 uint32_t K,
                                 const std::string &output_file)
   {
      float total_correct = 0;
      std::vector<float> query_recalls(num_queries, 0); // Recall for each query

      std::ofstream file(output_file);
      if (!file.is_open())
      {
         std::cerr << "Failed to open file: " << output_file << std::endl;
         return -1; // Return an error value if the file cannot be opened
      }

      file << "Query ID,Recall (%)\n"; // Write the CSV header

      for (uint32_t i = 0; i < num_queries; i++)
      {
         std::set<IdxType> gt_set;
         int32_t offset = -1;
         float correct_count = 0;

         for (uint32_t j = 0; j < K; j++)
         {
            if (gt[i * K + j].first != -1)
            {
               offset = j;
               gt_set.insert(gt[i * K + j].first);
            }
         }

         for (uint32_t j = 0; j < K; j++)
         {
            if (results[i * K + j].first == -1)
               break;
            if (offset >= 0 && results[i * K + j].second == gt[i * K + offset].second)
            { // Handle equal-cost cases
               correct_count++;
               offset--;
            }
            else
            {
               if (gt_set.find(results[i * K + j].first) != gt_set.end())
                  correct_count++;
            }
         }

         query_recalls[i] = correct_count / K; // Compute recall for the current query
         total_correct += correct_count;

         file << i << "," << query_recalls[i] << "\n"; // Write the record
      }

      file.close();

      return 100.0 * total_correct / (num_queries * K); // Return the overall recall
   }

   // fxy_add
   void save_roaring_vector(const std::string &filename, const std::vector<roaring::Roaring> &rb_vec)
   {
      std::ofstream out(filename, std::ios::binary);

      // Write the vector size
      uint64_t size = rb_vec.size();
      out.write(reinterpret_cast<const char *>(&size), sizeof(size));

      for (const auto &rb : rb_vec)
      {
         // Get the serialized size
         size_t serialized_size = rb.getSizeInBytes();
         char *buffer = new char[serialized_size];
         rb.write(buffer); // Write into the buffer

         // Write size followed by data
         out.write(reinterpret_cast<const char *>(&serialized_size), sizeof(serialized_size));
         out.write(buffer, serialized_size);

         delete[] buffer;
      }

      out.close();
      std::cout << "Saved roaring vector to " << filename << ", size = " << rb_vec.size() << std::endl;
   }

   void load_roaring_vector(const std::string &filename, std::vector<roaring::Roaring> &rb_vec)
   {
      std::ifstream in(filename, std::ios::binary);
      if (!in)
      {
         std::cerr << "Error: Could not open file for reading: " << filename << std::endl;
         return;
      }

      // Read the vector size
      uint64_t size;
      in.read(reinterpret_cast<char *>(&size), sizeof(size));
      rb_vec.resize(size);

      for (size_t i = 0; i < size; ++i)
      {
         // Read the size of each Roaring bitmap
         size_t serialized_size;
         in.read(reinterpret_cast<char *>(&serialized_size), sizeof(serialized_size));

         // Allocate the buffer and read the data
         char *buffer = new char[serialized_size];
         in.read(buffer, serialized_size);

         // Construct the Roaring bitmap
         rb_vec[i] = roaring::Roaring::readSafe(buffer, serialized_size);

         delete[] buffer;
      }

      in.close();
      std::cout << "Loaded roaring vector from " << filename << ", size = " << rb_vec.size() << std::endl;
   }

   // fxy_add helper: write vector data to a .fvecs file
   void write_fvecs(const std::string &filename, const std::vector<float *> &vecs, size_t dim)
   {
      std::ofstream out(filename, std::ios::binary);
      if (!out)
      {
         throw std::runtime_error("Cannot open file for writing: " + filename);
      }
      for (const auto &vec : vecs)
      {
         out.write(reinterpret_cast<const char *>(&dim), sizeof(uint32_t));
         out.write(reinterpret_cast<const char *>(vec), dim * sizeof(float));
      }
   }

   // fxy_add helper: write label sets to a .txt file in label1,label2,label3 format
   void write_labels_txt(const std::string &filename, const std::vector<std::vector<ANNS::LabelType>> &labels)
   {
      std::ofstream out(filename);
      if (!out)
      {
         throw std::runtime_error("Cannot open file for writing: " + filename);
      }
      for (const auto &label_set : labels)
      {
         for (size_t i = 0; i < label_set.size(); ++i)
         {
            out << label_set[i] << (i == label_set.size() - 1 ? "" : ",");
         }
         out << "\n";
      }
   }


   // fxxy_add: Generate a vector filter map for one query using a 2D-array inverted index, as required by NaviX
   std::vector<char> generate_single_filter_map(
      const std::vector<std::vector<ANNS::IdxType>>& inverted_index, 
      size_t N,                            
      const std::vector<uint32_t>& query_attrs) 
   {
      std::vector<char> filter_map(N, 0);
      size_t query_attr_count = query_attrs.size();

      if (query_attr_count == 0) {
         return filter_map; 
      }

      std::vector<int> match_counters(N, 0);
      std::vector<int> touched_indices;
      touched_indices.reserve(1024); 

      bool is_possible = true;
      for (uint32_t attr : query_attrs) {
         // Direct array indexing is faster than unordered_map::find()
         // Guard against query attributes that exceed the base inverted-index range
         if (attr < inverted_index.size() && !inverted_index[attr].empty()) {
            for (ANNS::IdxType xb_idx : inverted_index[attr]) {
               if (match_counters[xb_idx] == 0) {
                  touched_indices.push_back(xb_idx);
               }
               match_counters[xb_idx]++;
            }
         } else {
            // If a required query attribute is absent from the base set, no match is possible
            is_possible = false;
            break;
         }
      }
      
      if (is_possible) {
         for (int xb_idx : touched_indices) {
            if (match_counters[xb_idx] == query_attr_count) {
               filter_map[xb_idx] = 1;
            }
         }
      }
      return filter_map; // std::vector<char>: length-N filter array, where 1 means all query predicates are satisfied and 0 means they are not.
   }
}
