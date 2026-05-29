#ifndef ACORN_RABITQ_SIDE_INDEX_ACORN_H
#define ACORN_RABITQ_SIDE_INDEX_ACORN_H

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#ifndef ACORN_ENABLE_RABITQ
#define ACORN_ENABLE_RABITQ 0
#endif

#if ACORN_ENABLE_RABITQ
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/utils/rotator.hpp"
#endif

namespace acorn_rabitq {

class RabitQSideIndex {
   public:
    struct InitTiming {
        double rotate_ms = 0.0;
        double q_to_centroids_ms = 0.0;
        double wrapper_ms = 0.0;
    };

    struct QueryContext {
#if ACORN_ENABLE_RABITQ
        std::vector<float> rotated_query;
        std::vector<float> q_to_centroids;
        std::unique_ptr<rabitqlib::SplitSingleQuery<float>> query_wrapper;
#endif
    };

    RabitQSideIndex() = default;
    ~RabitQSideIndex() = default;

#if ACORN_ENABLE_RABITQ
    bool build(
        const float* base_vectors,
        std::size_t num_points,
        std::size_t dim,
        const std::vector<std::uint32_t>& point_to_group,
        std::uint32_t num_groups,
        std::size_t total_bits
    );

    bool save(const std::string& filename) const;
    bool load(const std::string& filename);
    std::uint64_t estimated_memory_bytes(bool include_rotator_state = false) const;
    std::uint64_t estimated_rotator_state_bytes() const;

    bool init_query(const float* query, QueryContext& ctx, InitTiming* timing = nullptr) const;
    float estimate_bin(std::uint32_t point_id, QueryContext& ctx, float* low_dist = nullptr) const;
    float estimate_full(std::uint32_t point_id, QueryContext& ctx, float* low_dist = nullptr) const;

    bool enabled() const { return enabled_; }
    std::size_t total_bits() const { return total_bits_; }
#else
    bool build(
        const float*,
        std::size_t,
        std::size_t,
        const std::vector<std::uint32_t>&,
        std::uint32_t,
        std::size_t
    ) { return false; }

    bool save(const std::string&) const { return false; }
    bool load(const std::string&) { return false; }
    std::uint64_t estimated_memory_bytes(bool include_rotator_state = false) const
    {
        (void)include_rotator_state;
        return 0;
    }
    std::uint64_t estimated_rotator_state_bytes() const { return 0; }

    bool init_query(const float*, QueryContext&, InitTiming* timing = nullptr) const
    {
        if (timing != nullptr)
        {
            *timing = InitTiming{};
        }
        return false;
    }
    float estimate_bin(std::uint32_t, QueryContext&, float* low_dist = nullptr) const
    {
        if (low_dist != nullptr)
        {
            *low_dist = 0.0f;
        }
        return 0.0f;
    }
    float estimate_full(std::uint32_t, QueryContext&, float* low_dist = nullptr) const
    {
        if (low_dist != nullptr)
        {
            *low_dist = 0.0f;
        }
        return 0.0f;
    }

    bool enabled() const { return false; }
    std::size_t total_bits() const { return 0; }
#endif

   private:
#if ACORN_ENABLE_RABITQ
    bool enabled_ = false;

    std::size_t num_points_ = 0;
    std::size_t dim_ = 0;
    std::size_t padded_dim_ = 0;
    std::size_t num_clusters_ = 0;
    std::size_t total_bits_ = 0;
    std::size_t ex_bits_ = 0;

    std::size_t size_bin_data_ = 0;
    std::size_t size_ex_data_ = 0;

    std::unique_ptr<rabitqlib::Rotator<float>> rotator_;
    rabitqlib::quant::RabitqConfig build_config_;
    rabitqlib::quant::RabitqConfig query_config_;
    float (*ip_func_)(const float*, const uint8_t*, std::size_t) = nullptr;

    std::vector<float> centroids_;
    std::vector<std::uint32_t> cluster_ids_;
    std::vector<std::uint8_t> bin_data_;
    std::vector<std::uint8_t> ex_data_;
#endif
};

}  // namespace acorn_rabitq

#endif  // ACORN_RABITQ_SIDE_INDEX_ACORN_H
