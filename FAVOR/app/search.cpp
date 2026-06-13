#include <sys/time.h>
#include <omp.h>

#include "dataset.h"
#include "favor.h"
#include "transform.h"

std::vector<std::vector<int>> load_groundtruth(const std::string &file_path, int k)
{
    std::ifstream reader(file_path, std::ios::binary | std::ios::ate);
    if (!reader)
    {
        throw std::runtime_error("Cannot open file: " + file_path);
    }

    std::streamsize file_size = reader.tellg();
    reader.seekg(0, std::ios::beg);

    if (file_size % sizeof(int) != 0)
    {
        throw std::runtime_error("File size is not a multiple of integer size");
    }

    size_t total_ints = file_size / sizeof(int);
    if (k <= 0 || total_ints % k != 0)
    {
        throw std::runtime_error("Invalid k value or file size mismatch");
    }

    std::vector<int> flat_data(total_ints);
    if (!reader.read(reinterpret_cast<char *>(flat_data.data()), file_size))
    {
        throw std::runtime_error("Failed to read file content");
    }

    size_t num_vectors = total_ints / k;
    std::vector<std::vector<int>> groundtruth;
    groundtruth.reserve(num_vectors);

    auto data_it = flat_data.begin();
    for (size_t i = 0; i < num_vectors; ++i)
    {
        groundtruth.emplace_back(data_it, data_it + k);
        data_it += k;
    }

    return groundtruth;
}

float interval(timeval &begin, timeval &end)
{
    return end.tv_sec - begin.tv_sec + (end.tv_usec - begin.tv_usec) * 1.0 / CLOCKS_PER_SEC;
}

int main(int argc, char *argv[])
{

    if (argc < 8)
    {
        std::cerr << "Usage: " << argv[0] << " baseset_path" << " queryset_path" << " attribute_path" << "topk" << " groundtruth_path" << " condition_path" << " index_path" << " ef\n"
                  << "condition_path: path to file containing filtering conditions for each query\n";
        return 1;
    }

    std::string baseset_path(argv[1]);
    std::string queryset_path(argv[2]);
    std::string attribute_path(argv[3]);

    std::string topk(argv[4]);
    int k = std::stoi(topk);

    std::string groundtruth_path(argv[5]);
    std::vector<std::vector<int>> groundtruth = load_groundtruth(groundtruth_path, k);

    BaseSet baseset;
    QuerySet queryset;
    baseset.read_data(baseset_path);
    queryset.read_data(queryset_path);
    baseset.get_attribute(attribute_path);

    std::string condition_path = argv[6];
    queryset.load_filtering_conditions(condition_path);

    std::string index_path = argv[7];
    int dim = baseset.dim;
    int num = baseset.num;

    std::string ef_(argv[8]);
    int ef = std::stoi(ef_);

    hnswlib::L2Space space(dim);
    favor::FAVOR<float> *alg_hnsw = new favor::FAVOR<float>(&space, index_path, false, num, ef);

    timeval begin, end;
    gettimeofday(&begin, NULL);

    float correct = 0;

    for (int i = 0; i < queryset.num; i++)
    {
            std::vector<FilterConditionWithId> conditions_with_id = ConditionTrans(queryset.filtering_conditions[i], baseset.attribute_map);
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(queryset.vectors.at(i).data(), k, conditions_with_id);
            for (int j = 0; j < k; j++)
            {
                if (std::find(groundtruth[i].begin(), groundtruth[i].end(), result.top().second) != groundtruth[i].end())
                    correct++;

                result.pop();
            }
    }

    gettimeofday(&end, NULL);

    float total = interval(begin, end);
    float recall = correct / (queryset.num * k);
    float average = total / queryset.num;
    float QPS = queryset.num / total;
    std::cout << "recall = " << recall << std::endl;
    std::cout << "average latency = " << average * 1000 << "ms" << std::endl;
    std::cout << "QPS = " << QPS << std::endl;

    delete alg_hnsw;
    return 0;
}
