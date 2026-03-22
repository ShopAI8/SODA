#include <omp.h>
#include <fstream>
#include <string>
#include <cstring>
#include <iostream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include "utils.h"
#include "storage.h"

namespace ANNS
{

   // claim the class
   template class Storage<float>;
   template class Storage<int8_t>;
   template class Storage<uint8_t>;

   // obtain the corresponding storage class
   std::shared_ptr<IStorage> create_storage(const std::string &data_type, bool verbose)
   {
      if (data_type == "float")
         return std::make_shared<Storage<float>>(DataType::FLOAT, verbose);
      else if (data_type == "int8")
         return std::make_shared<Storage<int8_t>>(DataType::INT8, verbose);
      else if (data_type == "uint8")
         return std::make_shared<Storage<uint8_t>>(DataType::UINT8, verbose);
      else
      {
         std::cerr << "Error: invalid data type " << data_type << std::endl;
         exit(-1);
      }
   }

   // obtain the corresponding storage class
   std::shared_ptr<IStorage> create_storage(std::shared_ptr<IStorage> storage, IdxType start, IdxType end)
   {
      DataType data_type = storage->get_data_type();
      if (data_type == DataType::FLOAT)
         return std::make_shared<Storage<float>>(storage, start, end);
      else if (data_type == DataType::INT8)
         return std::make_shared<Storage<int8_t>>(storage, start, end);
      else if (data_type == DataType::UINT8)
         return std::make_shared<Storage<uint8_t>>(storage, start, end);
      else
      {
         std::cerr << "Error: invalid data type " << data_type << std::endl;
         exit(-1);
      }
   }

   // construct the class
   template <typename T>
   Storage<T>::Storage(DataType data_type, bool verbose)
   {
      this->data_type = data_type;
      this->verbose = verbose;
   }

   // load from storage without copying data
   template <typename T>
   Storage<T>::Storage(std::shared_ptr<IStorage> storage, IdxType start, IdxType end)
   {
      data_type = storage->get_data_type();
      num_points = end - start;
      dim = storage->get_dim();
      vecs = reinterpret_cast<T *>(storage->get_vector(start));
      label_sets = storage->get_offseted_label_sets(start);
      prefetch_byte_num = dim * sizeof(T);
      verbose = false;
   }

   // 原始 load data
   /*template <typename T>
   void Storage<T>::load_from_file(const std::string &bin_file, const std::string &label_file, IdxType max_num_points)
   {
      if (verbose)
         std::cout << "Loading data from " << bin_file << " and " << label_file << " ..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();

      // open the binary file
      std::ifstream file(bin_file, std::ios::binary | std::ios::ate);
      if (!file.is_open())
         throw std::runtime_error("Failed to open file: " + bin_file);

      // read vector data
      file.seekg(0, std::ios::beg);
      file.read((char *)&num_points, sizeof(IdxType));
      file.read((char *)&dim, sizeof(IdxType));
      num_points = std::min(num_points, max_num_points);
      //vecs = static_cast<T *>(std::aligned_alloc(32, num_points * dim * sizeof(T)));
      //file.read((char *)vecs, num_points * dim * sizeof(T));
      vecs = static_cast<T *>(std::aligned_alloc(32, (size_t)num_points * dim * sizeof(T)));
      file.read((char *)vecs, (size_t)num_points * dim * sizeof(T)); //fxy_changed
      file.close();

      // for prefetch
      prefetch_byte_num = dim * sizeof(T);

      // read label data if exists
      std::map<LabelType, IdxType> label_cnts;
      label_sets = new std::vector<LabelType>[num_points];
      file.open(label_file);
      if (file.is_open())
      {
         std::string line, label;
         for (auto i = 0; i < num_points && i < max_num_points; ++i)
         {
            std::getline(file, line);
            std::stringstream ss(line);
            while (std::getline(ss, label, ','))
            {
               label_sets[i].emplace_back(std::stoi(label));
               if (label_cnts.find(std::stoi(label)) == label_cnts.end())
                  label_cnts[std::stoi(label)] = 1;
               else
                  label_cnts[std::stoi(label)]++;
            }
            std::sort(label_sets[i].begin(), label_sets[i].end());
            label_sets[i].shrink_to_fit();
         }
         file.close();

         // unfiltered ANNS when label file not found
      }
      else
      {
         std::cout << "- Warning: label file not found, set all labels to 1" << std::endl;
         for (auto i = 0; i < num_points && i < max_num_points; ++i)
            label_sets[i] = {1};
         label_cnts[1] = num_points;
      }

      // statistics
      if (verbose)
      {
         std::cout << "- Number of points: " << num_points << std::endl;
         std::cout << "- Dimension: " << dim << std::endl;
         std::cout << "- Number of labels: " << label_cnts.size() << std::endl;
         std::cout << "- Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << " ms" << std::endl;
         std::cout << SEP_LINE;
      }
   }*/

   // 原始 write data
   /*template <typename T>
   void Storage<T>::write_to_file(const std::string &bin_file, const std::string &label_file)
   {
      std::ofstream file(bin_file, std::ios::binary);
      file.write((char *)&num_points, sizeof(IdxType));
      file.write((char *)&dim, sizeof(IdxType));
      file.write((char *)vecs, num_points * dim * sizeof(T));
      file.close();

      // write label data
      file.open(label_file);
      if (!file.is_open())
      {
         std::cerr << "Error: failed to open file " << label_file << std::endl;
         return;
      }
      for (auto i = 0; i < num_points; ++i)
      {
         int size = label_sets[i].size();
         if (size == 0) // 检查是否为空
         {
            continue;
         }
         file << label_sets[i][0];
         for (auto j = 1; j < size; ++j)
            file << "," << label_sets[i][j];
         file << std::endl;
      }
      file.close();
   }*/
   
   //fxy_add 
   template <typename T>
   void Storage<T>::write_to_file(const std::string &bin_file, const std::string &label_file)
   {
      /**
       * 1.一句话概括函数核心作用：将内存中的向量数据和标签集持久化保存到磁盘。
       * 2.思路说明：
       * - 打开二进制文件，先写入点数（num_points）和维度（dim）。
       * - 计算总字节数时强制转换为 size_t，彻底避免 32 位整数溢出。
       * - 使用 1GB 的块（Chunk）分批写入向量数据，防止超大文件写入被操作系统或底层调用截断。
       * - 打开文本文件，逐行写入各向量对应的标签集（支持空标签行的安全写入）。
       */
      std::ofstream file(bin_file, std::ios::binary);
      if (!file.is_open())
      {
         std::cerr << "Error: failed to open file " << bin_file << std::endl;
         return;
      }

      file.write((char *)&num_points, sizeof(IdxType));
      file.write((char *)&dim, sizeof(IdxType));

      // 核心修复：转换为 size_t 并分块写入，防止大规模数据溢出和系统调用截断
      size_t total_bytes = (size_t)num_points * (size_t)dim * sizeof(T);
      size_t bytes_written = 0;
      const char* ptr = reinterpret_cast<const char*>(vecs);
      size_t chunk_size = 1024ULL * 1024ULL * 1024ULL; // 1GB 块

      while (bytes_written < total_bytes) {
         size_t chunk = std::min(total_bytes - bytes_written, chunk_size);
         file.write(ptr + bytes_written, chunk);
         bytes_written += chunk;
      }
      file.close();

      // write label data
      file.open(label_file);
      if (!file.is_open())
      {
         std::cerr << "Error: failed to open file " << label_file << std::endl;
         return;
      }
      for (auto i = 0; i < num_points; ++i)
      {
         int size = label_sets[i].size();
         if (size == 0) // 检查是否为空
         {
            file << std::endl; // 写入空行以维持 ID 对齐
            continue;
         }
         file << label_sets[i][0];
         for (auto j = 1; j < size; ++j)
            file << "," << label_sets[i][j];
         file << std::endl;
      }
      file.close();
   }


   // fxy_add 
   template <typename T>
   void Storage<T>::load_from_file(const std::string &bin_file, const std::string &label_file, IdxType max_num_points)
   {
      /**
       * 1.一句话概括函数核心作用：从磁盘读取大规模向量数据和对应的标签集到内存中。
       * 2.思路说明：
       * - 打开二进制文件读取头部，获取总点数和维度，并结合 max_num_points 确定实际读取量。
       * - 使用 std::aligned_alloc 为向量数据分配内存，确保数据对齐以加速后续计算。
       * - 计算总字节数时强制使用 size_t，并通过 1GB 分块（Chunk）安全读取，避免内存映射失败。
       * - 解析标签文本文件，将字符串转换为标签数组并排序，若文件不存在则设置默认标签。
       */
      if (verbose)
         std::cout << "Loading data from " << bin_file << " and " << label_file << " ..." << std::endl;
      auto start_time = std::chrono::high_resolution_clock::now();

      // open the binary file
      std::ifstream file(bin_file, std::ios::binary | std::ios::ate);
      if (!file.is_open())
         throw std::runtime_error("Failed to open file: " + bin_file);

      // read vector data
      file.seekg(0, std::ios::beg);
      file.read((char *)&num_points, sizeof(IdxType));
      file.read((char *)&dim, sizeof(IdxType));
      num_points = std::min(num_points, max_num_points);
      
      // 分配内存
      vecs = static_cast<T *>(std::aligned_alloc(32, (size_t)num_points * (size_t)dim * sizeof(T)));
      
      // 核心修复：转换为 size_t 并分块读取，防止 std::ifstream::read 一次性读取过大导致失败
      size_t total_bytes = (size_t)num_points * (size_t)dim * sizeof(T);
      size_t bytes_read = 0;
      char* ptr = reinterpret_cast<char*>(vecs);
      size_t chunk_size = 1024ULL * 1024ULL * 1024ULL; // 1GB 块

      while (bytes_read < total_bytes) {
         size_t chunk = std::min(total_bytes - bytes_read, chunk_size);
         file.read(ptr + bytes_read, chunk);
         if (!file) {
            std::cerr << "Warning: Reached EOF prematurely! Only read " << bytes_read << " bytes." << std::endl;
            break;
         }
         bytes_read += chunk;
      }
      file.close();

      // for prefetch
      prefetch_byte_num = dim * sizeof(T);

      // read label data if exists
      std::map<LabelType, IdxType> label_cnts;
      label_sets = new std::vector<LabelType>[num_points];
      file.open(label_file);
      if (file.is_open())
      {
         std::string line, label;
         for (auto i = 0; i < num_points && i < max_num_points; ++i)
         {
            std::getline(file, line);
            if (line.empty()) continue; // 处理空行
            
            std::stringstream ss(line);
            while (std::getline(ss, label, ','))
            {
               if (label.empty()) continue; // 防止空字符串引发 stoi 报错
               LabelType lbl = std::stoi(label);
               label_sets[i].emplace_back(lbl);
               if (label_cnts.find(lbl) == label_cnts.end())
                  label_cnts[lbl] = 1;
               else
                  label_cnts[lbl]++;
            }
            std::sort(label_sets[i].begin(), label_sets[i].end());
            label_sets[i].shrink_to_fit();
         }
         file.close();

         // unfiltered ANNS when label file not found
      }
      else
      {
         std::cout << "- Warning: label file not found, set all labels to 1" << std::endl;
         for (auto i = 0; i < num_points && i < max_num_points; ++i)
            label_sets[i] = {1};
         label_cnts[1] = num_points;
      }

      // statistics
      if (verbose)
      {
         std::cout << "- Number of points: " << num_points << std::endl;
         std::cout << "- Dimension: " << dim << std::endl;
         std::cout << "- Number of labels: " << label_cnts.size() << std::endl;
         std::cout << "- Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() << " ms" << std::endl;
         std::cout << "------------------------------------------------------------" << std::endl;
      }
   }

   // reorder the vector data
   template <typename T>
   void Storage<T>::reorder_data(const std::vector<IdxType> &new_to_old_ids)
   {

      // move the vectors and labels
      //auto new_vecs = static_cast<T *>(std::aligned_alloc(32, num_points * dim * sizeof(T)));
      auto new_vecs = static_cast<T *>(std::aligned_alloc(32, (size_t)num_points * dim * sizeof(T)));//fxy_changed
      auto new_label_sets = new std::vector<LabelType>[num_points];
      for (auto i = 0; i < num_points; ++i)
      {
         //std::memcpy(new_vecs + i * dim, vecs + new_to_old_ids[i] * dim, dim * sizeof(T));
         std::memcpy(new_vecs + (size_t)i * dim, vecs + (size_t)new_to_old_ids[i] * dim, (size_t)dim * sizeof(T)); //fxy_changed
         new_label_sets[i] = label_sets[new_to_old_ids[i]];
      }

      // clean up
      delete[] vecs;
      delete[] label_sets;
      vecs = new_vecs;
      label_sets = new_label_sets;
   }

   // obtain a point cloest to the center
   template <typename T>
   IdxType Storage<T>::choose_medoid(uint32_t num_threads, std::shared_ptr<DistanceHandler> distance_handler)
   {

      // compute center
      T *center = new T[dim]();
      for (auto id = 0; id < num_points; ++id)
         for (auto d = 0; d < dim; ++d)
            center[d] += *(vecs + id * dim + d);
      for (auto d = 0; d < dim; ++d)
         center[d] /= num_points;

      // obtain the closet point to the center
      std::vector<float> dists(num_points);
      omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 2048)
      for (auto id = 0; id < num_points; ++id)
         dists[id] = distance_handler->compute((const char *)center, get_vector(id), dim);
      IdxType medoid = std::min_element(dists.begin(), dists.end()) - dists.begin();

      // clean up
      delete[] center;
      return medoid;
   }
}