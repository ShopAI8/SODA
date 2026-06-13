#include "filter_condition.h"

std::vector<FilterConditionWithId> ConditionTrans(
    const std::vector<FilterCondition>& conditions,
    const std::map<std::string, int>& attribute_map)
{
    std::vector<FilterConditionWithId> result;
    
    for (const auto& cond : conditions) {
        result.emplace_back(cond, attribute_map);
    }
    
    return result;
}