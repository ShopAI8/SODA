#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include "dataset.h"
#include "transform.h"

bool QuerySet::check_condition(float *attribute, const FilterCondition& condition, std::map<std::string, int> attribute_map) const
{
    auto it = attribute_map.find(condition.attribute_name);
    if (it == attribute_map.end()) return false;
    
    int attribute_id = it->second;
    float value = attribute[attribute_id];
    
    if (condition.op == "IN") {
        return condition.attribute_value.find(value) != condition.attribute_value.end();
    } 
    else if (condition.attribute_value.size() == 1) {
        float ref_value = *condition.attribute_value.begin();
        if (condition.op == "==") return value == ref_value;
        if (condition.op == "!=") return value != ref_value;
        if (condition.op == ">")  return value > ref_value;
        if (condition.op == "<")  return value < ref_value;
        if (condition.op == ">=") return value >= ref_value;
        if (condition.op == "<=") return value <= ref_value;
    }
    return false;
}

bool QuerySet::check_all_conditions(float* attribute, const std::vector<FilterCondition>& conditions, std::map<std::string, int> attribute_map) const
{
    for (const auto& cond : conditions) {
        if (!check_condition(attribute, cond, attribute_map)) {
            return false;
        }
    }
    return true;
}

void QuerySet::load_filtering_conditions(const std::string& condition_path)
{
    std::ifstream reader(condition_path);
    if (!reader) {
        throw std::runtime_error("Cannot open condition file: " + condition_path);
    }

    filtering_conditions.clear();
    filtering_conditions.resize(num);

    std::string line;
    int query_id = 0;
    
    while (std::getline(reader, line) && query_id < num) {
        if (line.empty()) continue;
        
        try {
            Tokenizer tokenizer(line);
            std::vector<Token> tokens = tokenizer.tokenize();
            Parser parser(tokens);
            std::unique_ptr<Condition> ast = parser.parse();
            ast->trans(filtering_conditions[query_id]);
            query_id++;
        } catch (const std::exception& e) {
            std::cerr << "Error parsing condition for query " << query_id << ": " << e.what() << std::endl;
            throw;
        }
    }

    reader.close();
    std::cout << "Loaded filtering conditions for " << query_id << " queries" << std::endl;
}