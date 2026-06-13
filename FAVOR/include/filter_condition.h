#pragma once

#include <set>
#include <string>
#include <map>
#include <vector>
#include <stdexcept>

struct FilterCondition {
    std::string attribute_name;        
    std::string op;
    std::set<float> attribute_value;

    FilterCondition(std::string a, std::string b, const std::set<float>& c) 
        : attribute_name(a), op(b), attribute_value(c) {}
    
    FilterCondition() = default;
};

struct FilterConditionWithId {
    int attribute_id;
    std::string op;
    std::set<float> attribute_value;

    FilterConditionWithId(
        const FilterCondition& condition,
        const std::map<std::string, int>& attribute_map
    ) : op(condition.op), attribute_value(condition.attribute_value) 
    {
        auto it = attribute_map.find(condition.attribute_name);
        if (it == attribute_map.end()) {
            throw std::invalid_argument("Attribute name not found in map: " + condition.attribute_name);
        }
        attribute_id = it->second;
    }

    FilterConditionWithId(int id, std::string o, const std::set<float>& vals)
        : attribute_id(id), op(std::move(o)), attribute_value(vals) {}

    FilterConditionWithId() = default;
};

std::vector<FilterConditionWithId> ConditionTrans(
    const std::vector<FilterCondition>& conditions,
    const std::map<std::string, int>& attribute_map);