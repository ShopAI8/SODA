#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

namespace fs = boost::filesystem;
namespace po = boost::program_options;

namespace {

std::map<std::string, std::string>
parse_meta_file(const std::string& meta_path) {
   std::ifstream in(meta_path);
   if (!in.is_open()) {
      throw std::runtime_error("failed to open meta file: " + meta_path);
   }

   std::map<std::string, std::string> kvs;
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

void
print_meta_summary(const std::string& title, const fs::path& meta_path) {
   std::cout << "=== " << title << " ===" << std::endl;
   std::cout << "meta: " << meta_path.string() << std::endl;
   const auto kvs = parse_meta_file(meta_path.string());
   for (const auto& kv : kvs) {
      std::cout << "  " << std::left << std::setw(18) << kv.first << " : " << kv.second << std::endl;
   }
   std::cout << std::endl;
}

}  // namespace

int
main(int argc, char** argv) {
   std::string index_root;

   try {
      po::options_description desc{"Arguments"};
      desc.add_options()
         ("help,h", "Print help")
         ("index_root", po::value<std::string>(&index_root)->required(),
          "Index root directory that contains Milvus-IVF/ and Milvus-HNSW/");

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

   try {
      const fs::path root(index_root);
      print_meta_summary("Milvus-IVF", root / "Milvus-IVF" / "milvus_knowhere.meta");
      print_meta_summary("Milvus-HNSW", root / "Milvus-HNSW" / "milvus_knowhere_hnsw.meta");
   } catch (const std::exception& ex) {
      std::cerr << ex.what() << std::endl;
      return 1;
   }

   return 0;
}
