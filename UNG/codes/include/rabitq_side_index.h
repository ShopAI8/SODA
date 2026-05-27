#ifndef ANNS_RABITQ_SIDE_INDEX_H
#define ANNS_RABITQ_SIDE_INDEX_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "config.h"
#include "storage.h"

#include "rabitqlib/index/query.hpp"
#include "rabitqlib/utils/rotator.hpp"

namespace ANNS::rabitq {

class RabitQSideIndex {
   public:
    struct InitTiming {
        double rotate_ms = 0.0;
        double q_to_centroids_ms = 0.0;
        double wrapper_ms = 0.0;
    };

    struct QueryContext {
        std::vector<float> rotated_query;
        std::vector<float> q_to_centroids;
        std::unique_ptr<rabitqlib::SplitSingleQuery<float>> query_wrapper;
    };

    RabitQSideIndex() = default;
    ~RabitQSideIndex() = default;

    bool build(
        const std::shared_ptr<ANNS::IStorage>& base_storage,
        const std::vector<ANNS::IdxType>& point_to_group,
        ANNS::IdxType num_groups,
        size_t total_bits
    );

    bool save(const std::string& filename) const;
    bool load(const std::string& filename);
    [[nodiscard]] uint64_t estimated_memory_bytes(bool include_rotator_state = false) const;
    [[nodiscard]] uint64_t estimated_rotator_state_bytes() const;

    bool init_query(const char* query, QueryContext& ctx, InitTiming* timing = nullptr) const;
    bool build_query_wrapper(QueryContext& ctx) const;
    bool save_query_context_cache(
        const std::string& filename,
        const std::vector<std::unique_ptr<QueryContext>>& cache
    ) const;
    bool load_query_context_cache(
        const std::string& filename,
        std::vector<std::unique_ptr<QueryContext>>& cache
    ) const;
    float estimate_bin(
        ANNS::IdxType point_id,
        QueryContext& ctx,
        float* low_dist = nullptr
    ) const;
    float estimate_full(
        ANNS::IdxType point_id,
        QueryContext& ctx,
        float* low_dist = nullptr
    ) const;

    [[nodiscard]] bool enabled() const { return enabled_; }
    [[nodiscard]] size_t total_bits() const { return total_bits_; }
    [[nodiscard]] size_t ex_bits() const { return ex_bits_; }

   private:
    bool enabled_ = false;

    size_t num_points_ = 0;
    size_t dim_ = 0;
    size_t padded_dim_ = 0;
    size_t num_clusters_ = 0;
    size_t total_bits_ = 0;
    size_t ex_bits_ = 0;

    size_t size_bin_data_ = 0;
    size_t size_ex_data_ = 0;

    std::unique_ptr<rabitqlib::Rotator<float>> rotator_;
    rabitqlib::quant::RabitqConfig build_config_;
    rabitqlib::quant::RabitqConfig query_config_;
    float (*ip_func_)(const float*, const uint8_t*, size_t) = nullptr;

    std::vector<float> centroids_;
    std::vector<ANNS::IdxType> cluster_ids_;
    std::vector<uint8_t> bin_data_;
    std::vector<uint8_t> ex_data_;
};

}  // namespace ANNS::rabitq

#endif  // ANNS_RABITQ_SIDE_INDEX_H
