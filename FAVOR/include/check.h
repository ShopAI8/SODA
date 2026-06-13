#pragma once

#include <functional>
#include <vector>
#include <set>
#include <string>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cmath>

#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
    #ifndef USE_AVX2
    #define USE_AVX2
    #endif
#endif

#include "filter_condition.h"

typedef float attributetype;

enum class ConditionOp : uint8_t
{
    IN,
    EQ,
    NE,
    GT,
    LT,
    GE,
    LE
};

namespace {
    namespace branchless {
        inline __attribute__((always_inline, hot)) bool checkEQ(attributetype value, attributetype ref) {
            return value == ref;
        }
        
        inline __attribute__((always_inline, hot)) bool checkNE(attributetype value, attributetype ref) {
            return value != ref;
        }
        
        inline __attribute__((always_inline, hot)) bool checkGT(attributetype value, attributetype ref) {
            return value > ref;
        }
        
        inline __attribute__((always_inline, hot)) bool checkLT(attributetype value, attributetype ref) {
            return value < ref;
        }
        
        inline __attribute__((always_inline, hot)) bool checkGE(attributetype value, attributetype ref) {
            return value >= ref;
        }
        
        inline __attribute__((always_inline, hot)) bool checkLE(attributetype value, attributetype ref) {
            return value <= ref;
        }
        
        inline __attribute__((always_inline, hot)) bool checkIN(attributetype value, const attributetype* values, uint32_t count) {
            if (count == 0) return false;
            for (uint32_t i = 0; i < count; ++i) {
                if (values[i] == value) return true;
            }
            return false;
        }
    }
}

struct alignas(16) FastCondition {
    uint32_t attribute_id;
    ConditionOp op;
    attributetype ref_value;
    uint32_t in_values_count;
    const attributetype* in_values_ptr;
    
    using CheckFunc = bool(*)(attributetype, attributetype);
    using CheckFuncIN = bool(*)(attributetype, const attributetype*, uint32_t);
    
    CheckFunc check_func_;
    CheckFuncIN check_func_in_;

    FastCondition() = default;

    explicit FastCondition(const FilterConditionWithId& cond, std::vector<attributetype>& in_vals_storage, size_t& storage_offset) {
        attribute_id = static_cast<uint32_t>(cond.attribute_id);
        op = getConditionOp(cond.op);

        if (op == ConditionOp::IN) {
            in_values_count = static_cast<uint32_t>(cond.attribute_value.size());
            if (in_values_count > 0) {
                in_values_ptr = &in_vals_storage[storage_offset];
                for (const auto& val : cond.attribute_value) {
                    in_vals_storage[storage_offset++] = val;
                }
            } else {
                in_values_ptr = nullptr;
            }
            ref_value = 0;
            check_func_ = nullptr;
            check_func_in_ = branchless::checkIN;
        } else {
            in_values_count = 0;
            in_values_ptr = nullptr;
            ref_value = cond.attribute_value.empty() ? 0 : *cond.attribute_value.begin();
            check_func_in_ = nullptr;
            
            switch (op) {
                case ConditionOp::EQ: check_func_ = branchless::checkEQ; break;
                case ConditionOp::NE: check_func_ = branchless::checkNE; break;
                case ConditionOp::GT: check_func_ = branchless::checkGT; break;
                case ConditionOp::LT: check_func_ = branchless::checkLT; break;
                case ConditionOp::GE: check_func_ = branchless::checkGE; break;
                case ConditionOp::LE: check_func_ = branchless::checkLE; break;
                default: check_func_ = branchless::checkEQ; break;
            }
        }
    }

    inline __attribute__((always_inline, hot)) bool check(const attributetype* attribute) const {
        attributetype value = attribute[attribute_id];
        
        if (op == ConditionOp::IN) {
            return check_func_in_(value, in_values_ptr, in_values_count);
        } else {
            return check_func_(value, ref_value);
        }
    }

private:
    static ConditionOp getConditionOp(const std::string &opStr) {
        if (opStr == "==") return ConditionOp::EQ;
        if (opStr == "<")  return ConditionOp::LT;
        if (opStr == ">")  return ConditionOp::GT;
        if (opStr == ">=") return ConditionOp::GE;
        if (opStr == "<=") return ConditionOp::LE;
        if (opStr == "!=") return ConditionOp::NE;
        if (opStr == "IN") return ConditionOp::IN;
        return ConditionOp::EQ;
    }
};

class OptimizedFilter
{
private:
    std::vector<FastCondition> conditions_;
    std::vector<attributetype> in_values_storage_;

public:
    OptimizedFilter() = default;

    OptimizedFilter(const std::vector<FilterConditionWithId> &filtering_conditions) {
        size_t total_in_values = 0;
        for (const auto &cond : filtering_conditions) {
            if (cond.op == "IN") {
                total_in_values += cond.attribute_value.size();
            }
        }

        in_values_storage_.reserve(total_in_values);
        conditions_.reserve(filtering_conditions.size());

        size_t storage_offset = 0;
        for (const auto &cond : filtering_conditions) {
            conditions_.emplace_back(cond, in_values_storage_, storage_offset);
        }
    }

    inline __attribute__((always_inline, hot)) bool check(const attributetype* attribute) const {
        for (const auto &condition : conditions_) {
            if (!condition.check(attribute)) [[unlikely]] {
                return false;
            }
        }
        return true;
    }

#ifdef USE_AVX2
    inline __attribute__((always_inline, hot)) void checkBatchAVX2(
        const char* data_level0_memory,
        size_t size_data_per_element,
        size_t attribute_offset,
        size_t start_idx,
        size_t end_idx,
        std::vector<size_t>& candidates) const
    {
        if (conditions_.empty()) {
            for (size_t i = start_idx; i < end_idx; i++) {
                candidates.push_back(i);
            }
            return;
        }

        const FastCondition& cond = conditions_[0];
        
        if (cond.op == ConditionOp::EQ && conditions_.size() == 1) {
            const float ref_value = cond.ref_value;
            const __m256 ref_vec = _mm256_set1_ps(ref_value);
            
            size_t i = start_idx;
            
            const size_t prefetch_distance = 64 / sizeof(attributetype);
            const size_t prefetch_distance2 = prefetch_distance * 4; 
            
            for (; i + 8 <= end_idx; i += 8) {
                if (i + prefetch_distance < end_idx) {
                    const char* prefetch_ptr = data_level0_memory + (i + prefetch_distance) * size_data_per_element + attribute_offset;
                    _mm_prefetch(prefetch_ptr, _MM_HINT_T0);
                }
                
                if (i + prefetch_distance2 < end_idx) {
                    const char* prefetch_ptr2 = data_level0_memory + (i + prefetch_distance2) * size_data_per_element + attribute_offset;
                    _mm_prefetch(prefetch_ptr2, _MM_HINT_T1);
                }
                
                float values[8];
                for (int j = 0; j < 8; j++) {
                    const char* element_ptr = data_level0_memory + (i + j) * size_data_per_element;
                    const attributetype* attr_ptr = reinterpret_cast<const attributetype*>(element_ptr + attribute_offset);
                    values[j] = *attr_ptr;
                }
                
                __m256 val_vec = _mm256_loadu_ps(values);
                __m256 cmp_result = _mm256_cmp_ps(val_vec, ref_vec, _CMP_EQ_OQ);
                int mask = _mm256_movemask_ps(cmp_result);
                
                for (int j = 0; j < 8; j++) {
                    if (mask & (1 << j)) [[unlikely]] {
                        candidates.push_back(i + j);
                    }
                }
            }
            
            for (; i < end_idx; i++) {
                const char* element_ptr = data_level0_memory + i * size_data_per_element;
                const attributetype* attr_ptr = reinterpret_cast<const attributetype*>(element_ptr + attribute_offset);
                if (check(attr_ptr)) [[unlikely]] {
                    candidates.push_back(i);
                }
            }
        } else {
            const size_t prefetch_distance = 16;
            const size_t prefetch_distance2 = 64;
            
            for (size_t i = start_idx; i < end_idx; i++) {
                if (i + prefetch_distance < end_idx) {
                    const char* prefetch_ptr = data_level0_memory + (i + prefetch_distance) * size_data_per_element + attribute_offset;
                    _mm_prefetch(prefetch_ptr, _MM_HINT_T0);
                }
                if (i + prefetch_distance2 < end_idx) {
                    const char* prefetch_ptr2 = data_level0_memory + (i + prefetch_distance2) * size_data_per_element + attribute_offset;
                    _mm_prefetch(prefetch_ptr2, _MM_HINT_T1);
                }
                
                const char* element_ptr = data_level0_memory + i * size_data_per_element;
                const attributetype* attr_ptr = reinterpret_cast<const attributetype*>(element_ptr + attribute_offset);
                if (check(attr_ptr)) [[unlikely]] {
                    candidates.push_back(i);
                }
            }
        }
    }
#endif

    size_t conditionCount() const {
        return conditions_.size();
    }

    std::string getCacheKey() const {
        std::string key;
        key.reserve(conditions_.size() * 16);
        
        for (const auto& cond : conditions_) {
            key += std::to_string(cond.attribute_id) + ":";
            key += std::to_string(static_cast<int>(cond.op)) + ":";
            
            if (cond.op == ConditionOp::IN) {
                key += "[";
                for (uint32_t i = 0; i < cond.in_values_count; i++) {
                    if (i > 0) key += ",";
                    key += std::to_string(cond.in_values_ptr[i]);
                }
                key += "]";
            } else {
                key += std::to_string(cond.ref_value);
            }
            key += "|";
        }
        
        return key;
    }
};
