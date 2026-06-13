#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <math.h>
#include <omp.h>

#include "dataset.h"
#include "transform.h"

inline bool distance_compare(const std::pair<int, float> &a, const std::pair<int, float> &b)
{
    return a.second < b.second;
}

float compute_distance_l2(std::vector<float> vector_1, std::vector<float> vector_2, int dim)
{
    float distance = 0;
    for (int i = 0; i < dim; i++)
    {
        float sub = vector_1[i] - vector_2[i];
        distance += sub * sub;
    }
    return distance;
}

int main(int argc, char *argv[])
{
    if (argc < 7)
    {
        std::cerr << "Usage: " << argv[0] << " baseset_path" << " queryset_path" << " attribute_path" << "topk" << " groundtruth_path" << " condition_path\n"
                  << "condition_path: path to file containing filtering conditions for each query\n";
        return 1;
    }

    std::string baseset_path(argv[1]);
    std::string queryset_path(argv[2]);
    std::string attribute_path(argv[3]);

    std::string topk(argv[4]);
    int k = std::stoi(topk);

    std::string groundtruth_path(argv[5]);

    BaseSet baseset;
    QuerySet queryset;
    baseset.read_data(baseset_path);
    queryset.read_data(queryset_path);
    baseset.get_attribute(attribute_path);

    std::string condition_path = argv[6];
    queryset.load_filtering_conditions(condition_path);

    std::cout << "begin computing groundtruth" << std::endl;

    std::vector<std::vector<int>> temp_groundtruth(queryset.num);
    std::vector<float> temp_distance(queryset.num);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < queryset.num; i++)
    {
        std::vector<std::pair<int, float>> gt_dist;
        for (int j = 0; j < baseset.num; j++)
        {
            if (!queryset.check_all_conditions(baseset.attribute + j * baseset.attribute_num, queryset.filtering_conditions[i], baseset.attribute_map))
            {
                continue;
            }
            float dist = compute_distance_l2(queryset.vectors[i], baseset.vectors[j], baseset.dim);
            gt_dist.emplace_back(std::make_pair(baseset.vector_id[j], dist));
        }
        while (gt_dist.size() < static_cast<size_t>(k))
        {
            gt_dist.emplace_back(std::make_pair(-1, 1000000));
        }
        std::sort(gt_dist.begin(), gt_dist.end(), distance_compare);

        temp_distance[i] = (gt_dist[k - 1].second - gt_dist[0].second) / (k - 1);

        std::vector<int> groundtruth;
        for (int j = 0; j < k; j++)
        {
            groundtruth.emplace_back(gt_dist[j].first);
        }

        temp_groundtruth[i] = groundtruth;
    }

    std::vector<std::vector<int>> groundtruth = temp_groundtruth;

    float total_distance = 0.0f;
    for (int i = 0; i < queryset.num; i++)
    {
        total_distance += temp_distance[i];
    }
    float avg_distance = total_distance / queryset.num;
    std::cout << "Average nearest distance: " << avg_distance << std::endl;

    std::cout << "begin writing to " << groundtruth_path << std::endl;
    std::ofstream writer(groundtruth_path, std::ios::binary);
    for (int i = 0; i < queryset.num; i++)
        for (int j = 0; j < k; j++)
            writer.write((char *)&groundtruth[i][j], sizeof(int));
    writer.close();
    std::cout << "Finish" << std::endl;

    return 0;
}