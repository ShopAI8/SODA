#include <string>
#include <vector>
#include <map>

#include "filter_condition.h"

class DataSet
{
public:
    int num;
    int dim;
    int attribute_num;
    std::vector<std::vector<float>> vectors;
    std::vector<int> vector_id;

    DataSet() {};

    ~DataSet() {}

    void read_data(const std::string &dataset_path);
};

class BaseSet : public DataSet
{
public:
    // attribute[vector_id][attribute_id] = attribute + (vector_id * attribute_num + attribute_id)
    float *attribute{nullptr};

    std::map<std::string, int> attribute_map;

    void get_attribute(const std::string &attribute_path);

    ~BaseSet() {
        if (attribute != nullptr) {
            delete[] attribute;
            attribute = nullptr;
        }
    }
};

class QuerySet : public DataSet
{
public:
    std::vector<std::vector<FilterCondition>> filtering_conditions;

    bool check_condition(float *attribute, const FilterCondition& condition, std::map<std::string, int> attribute_map) const;

    bool check_all_conditions(float *attribute, const std::vector<FilterCondition>& conditions, std::map<std::string, int> attribute_map) const;
    
    void load_filtering_conditions(const std::string& condition_path);
};