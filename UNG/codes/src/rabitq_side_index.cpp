#include "rabitq_side_index.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <chrono>
#include <vector>

#include "rabitqlib/index/estimator.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/utils/space.hpp"

namespace {

constexpr uint32_t kRabitQMagic = 0x51425452U;  // "RBTQ"
constexpr uint32_t kRabitQVersion = 1U;
constexpr uint32_t kRabitQQueryCtxMagic = 0x51435452U;  // "RTCQ"
constexpr uint32_t kRabitQQueryCtxVersion = 1U;

template <typename T>
bool read_binary(std::ifstream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    return in.good();
}

template <typename T>
bool write_binary(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    return out.good();
}

}  // namespace

namespace ANNS::rabitq {

bool RabitQSideIndex::build(
    const std::shared_ptr<ANNS::IStorage>& base_storage,
    const std::vector<ANNS::IdxType>& point_to_group,
    ANNS::IdxType num_groups,
    size_t total_bits
) {
    enabled_ = false;

    if (!base_storage) {
        std::cerr << "[RabitQ] build failed: null base storage.\n";
        return false;
    }
    if (base_storage->get_data_type() != ANNS::DataType::FLOAT) {
        std::cerr << "[RabitQ] build failed: only float data is supported.\n";
        return false;
    }
    if (total_bits < 1 || total_bits > 9) {
        std::cerr << "[RabitQ] build failed: total_bits must be in [1, 9].\n";
        return false;
    }

    num_points_ = static_cast<size_t>(base_storage->get_num_points());
    dim_ = static_cast<size_t>(base_storage->get_dim());
    num_clusters_ = static_cast<size_t>(num_groups);
    total_bits_ = total_bits;
    ex_bits_ = total_bits_ - 1;

    if (point_to_group.size() != num_points_) {
        std::cerr << "[RabitQ] build failed: point_to_group size mismatch.\n";
        return false;
    }
    if (num_clusters_ == 0) {
        std::cerr << "[RabitQ] build failed: num_clusters is zero.\n";
        return false;
    }

    rotator_.reset(rabitqlib::choose_rotator<float>(
        dim_,
        rabitqlib::RotatorType::FhtKacRotator,
        rabitqlib::round_up_to_multiple(dim_, 64)
    ));
    if (!rotator_) {
        std::cerr << "[RabitQ] build failed: cannot create rotator.\n";
        return false;
    }
    padded_dim_ = rotator_->size();
    if (padded_dim_ < dim_ || padded_dim_ % 64 != 0) {
        std::cerr << "[RabitQ] build failed: invalid padded dimension.\n";
        return false;
    }

    size_bin_data_ = rabitqlib::BinDataMap<float>::data_bytes(padded_dim_);
    size_ex_data_ = rabitqlib::ExDataMap<float>::data_bytes(padded_dim_, ex_bits_);

    centroids_.assign(num_clusters_ * padded_dim_, 0.0F);
    cluster_ids_.assign(num_points_, 0U);
    bin_data_.assign(num_points_ * size_bin_data_, 0U);
    ex_data_.assign(num_points_ * size_ex_data_, 0U);

    std::vector<size_t> cluster_counts(num_clusters_, 0U);
    std::vector<float> rotated(padded_dim_, 0.0F);

    for (size_t point_id = 0; point_id < num_points_; ++point_id) {
        ANNS::IdxType raw_group_id = point_to_group[point_id];
        if (raw_group_id == 0 || raw_group_id > num_groups) {
            std::cerr << "[RabitQ] build failed: invalid group id in point_to_group.\n";
            return false;
        }

        const size_t cluster_id = static_cast<size_t>(raw_group_id - 1);
        cluster_ids_[point_id] = static_cast<ANNS::IdxType>(cluster_id);
        cluster_counts[cluster_id] += 1U;

        const char* vec_ptr = base_storage->get_vector(static_cast<ANNS::IdxType>(point_id));
        rotator_->rotate(reinterpret_cast<const float*>(vec_ptr), rotated.data());

        float* centroid = centroids_.data() + cluster_id * padded_dim_;
        for (size_t d = 0; d < padded_dim_; ++d) {
            centroid[d] += rotated[d];
        }
    }

    for (size_t cluster_id = 0; cluster_id < num_clusters_; ++cluster_id) {
        float* centroid = centroids_.data() + cluster_id * padded_dim_;
        const size_t count = cluster_counts[cluster_id];
        if (count == 0U) {
            continue;
        }
        const float inv_count = 1.0F / static_cast<float>(count);
        for (size_t d = 0; d < padded_dim_; ++d) {
            centroid[d] *= inv_count;
        }
    }

    build_config_ = rabitqlib::quant::faster_config(padded_dim_, ex_bits_ + 1);
    query_config_ =
        rabitqlib::quant::faster_config(padded_dim_, rabitqlib::SplitSingleQuery<float>::kNumBits);
    ip_func_ = rabitqlib::select_excode_ipfunc(ex_bits_);

    for (size_t point_id = 0; point_id < num_points_; ++point_id) {
        const size_t cluster_id = static_cast<size_t>(cluster_ids_[point_id]);
        const char* vec_ptr = base_storage->get_vector(static_cast<ANNS::IdxType>(point_id));
        rotator_->rotate(reinterpret_cast<const float*>(vec_ptr), rotated.data());

        char* bin_ptr = reinterpret_cast<char*>(bin_data_.data() + point_id * size_bin_data_);
        char* ex_ptr = reinterpret_cast<char*>(ex_data_.data() + point_id * size_ex_data_);
        const float* centroid = centroids_.data() + cluster_id * padded_dim_;

        rabitqlib::quant::quantize_split_single(
            rotated.data(),
            centroid,
            padded_dim_,
            ex_bits_,
            bin_ptr,
            ex_ptr,
            rabitqlib::METRIC_L2,
            build_config_
        );
    }

    enabled_ = true;
    return true;
}

bool RabitQSideIndex::save(const std::string& filename) const {
    if (!enabled_ || !rotator_) {
        std::cerr << "[RabitQ] save skipped: side index is not enabled.\n";
        return false;
    }

    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "[RabitQ] save failed: cannot open " << filename << '\n';
        return false;
    }

    if (!write_binary(out, kRabitQMagic) || !write_binary(out, kRabitQVersion) ||
        !write_binary(out, num_points_) || !write_binary(out, dim_) ||
        !write_binary(out, padded_dim_) || !write_binary(out, num_clusters_) ||
        !write_binary(out, total_bits_) || !write_binary(out, ex_bits_) ||
        !write_binary(out, size_bin_data_) || !write_binary(out, size_ex_data_)) {
        std::cerr << "[RabitQ] save failed: failed to write header.\n";
        return false;
    }

    out.write(
        reinterpret_cast<const char*>(cluster_ids_.data()),
        static_cast<std::streamsize>(cluster_ids_.size() * sizeof(ANNS::IdxType))
    );
    out.write(
        reinterpret_cast<const char*>(centroids_.data()),
        static_cast<std::streamsize>(centroids_.size() * sizeof(float))
    );
    out.write(
        reinterpret_cast<const char*>(bin_data_.data()),
        static_cast<std::streamsize>(bin_data_.size() * sizeof(uint8_t))
    );
    out.write(
        reinterpret_cast<const char*>(ex_data_.data()),
        static_cast<std::streamsize>(ex_data_.size() * sizeof(uint8_t))
    );

    if (!out.good()) {
        std::cerr << "[RabitQ] save failed: failed to write payload.\n";
        return false;
    }

    rotator_->save(out);
    if (!out.good()) {
        std::cerr << "[RabitQ] save failed: failed to write rotator state.\n";
        return false;
    }

    return true;
}

bool RabitQSideIndex::load(const std::string& filename) {
    enabled_ = false;
    rotator_.reset();

    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "[RabitQ] load failed: cannot open " << filename << '\n';
        return false;
    }

    uint32_t magic = 0;
    uint32_t version = 0;
    if (!read_binary(in, magic) || !read_binary(in, version)) {
        std::cerr << "[RabitQ] load failed: cannot read header.\n";
        return false;
    }
    if (magic != kRabitQMagic || version != kRabitQVersion) {
        std::cerr << "[RabitQ] load failed: incompatible side index format.\n";
        return false;
    }

    if (!read_binary(in, num_points_) || !read_binary(in, dim_) ||
        !read_binary(in, padded_dim_) || !read_binary(in, num_clusters_) ||
        !read_binary(in, total_bits_) || !read_binary(in, ex_bits_) ||
        !read_binary(in, size_bin_data_) || !read_binary(in, size_ex_data_)) {
        std::cerr << "[RabitQ] load failed: cannot read metadata.\n";
        return false;
    }

    if (num_points_ == 0 || num_clusters_ == 0 || padded_dim_ % 64 != 0) {
        std::cerr << "[RabitQ] load failed: invalid metadata values.\n";
        return false;
    }

    cluster_ids_.assign(num_points_, 0U);
    centroids_.assign(num_clusters_ * padded_dim_, 0.0F);
    bin_data_.assign(num_points_ * size_bin_data_, 0U);
    ex_data_.assign(num_points_ * size_ex_data_, 0U);

    in.read(
        reinterpret_cast<char*>(cluster_ids_.data()),
        static_cast<std::streamsize>(cluster_ids_.size() * sizeof(ANNS::IdxType))
    );
    in.read(
        reinterpret_cast<char*>(centroids_.data()),
        static_cast<std::streamsize>(centroids_.size() * sizeof(float))
    );
    in.read(
        reinterpret_cast<char*>(bin_data_.data()),
        static_cast<std::streamsize>(bin_data_.size() * sizeof(uint8_t))
    );
    in.read(
        reinterpret_cast<char*>(ex_data_.data()),
        static_cast<std::streamsize>(ex_data_.size() * sizeof(uint8_t))
    );
    if (!in.good()) {
        std::cerr << "[RabitQ] load failed: cannot read payload.\n";
        return false;
    }

    rotator_.reset(rabitqlib::choose_rotator<float>(
        dim_,
        rabitqlib::RotatorType::FhtKacRotator,
        rabitqlib::round_up_to_multiple(dim_, 64)
    ));
    if (!rotator_ || rotator_->size() != padded_dim_) {
        std::cerr << "[RabitQ] load failed: invalid rotator state.\n";
        return false;
    }
    rotator_->load(in);
    if (!in.good()) {
        std::cerr << "[RabitQ] load failed: cannot read rotator state.\n";
        return false;
    }

    build_config_ = rabitqlib::quant::faster_config(padded_dim_, ex_bits_ + 1);
    query_config_ =
        rabitqlib::quant::faster_config(padded_dim_, rabitqlib::SplitSingleQuery<float>::kNumBits);
    ip_func_ = rabitqlib::select_excode_ipfunc(ex_bits_);

    enabled_ = true;
    return true;
}

uint64_t RabitQSideIndex::estimated_rotator_state_bytes() const {
    // Keep estimator strictly sizeof/data-buffer based for consistency with
    // existing index_size accounting style.
    return 0;
}

uint64_t RabitQSideIndex::estimated_memory_bytes(bool include_rotator_state) const {
    if (!enabled_) {
        return 0;
    }

    uint64_t bytes = 0;
    bytes += static_cast<uint64_t>(cluster_ids_.size()) * sizeof(ANNS::IdxType);
    bytes += static_cast<uint64_t>(centroids_.size()) * sizeof(float);
    bytes += static_cast<uint64_t>(bin_data_.size()) * sizeof(uint8_t);
    bytes += static_cast<uint64_t>(ex_data_.size()) * sizeof(uint8_t);

    if (include_rotator_state) {
        bytes += estimated_rotator_state_bytes();
    }
    return bytes;
}

bool RabitQSideIndex::init_query(const char* query, QueryContext& ctx, InitTiming* timing) const {
    if (!enabled_ || !rotator_ || query == nullptr) {
        return false;
    }

    if (timing) {
        *timing = InitTiming{};
    }

    auto rotate_start = std::chrono::high_resolution_clock::now();
    ctx.rotated_query.assign(padded_dim_, 0.0F);
    rotator_->rotate(reinterpret_cast<const float*>(query), ctx.rotated_query.data());
    if (timing) {
        timing->rotate_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - rotate_start
        ).count();
    }

    auto wrapper_start = std::chrono::high_resolution_clock::now();
    if (!build_query_wrapper(ctx)) {
        return false;
    }
    if (timing) {
        timing->wrapper_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - wrapper_start
        ).count();
    }

    auto q2c_start = std::chrono::high_resolution_clock::now();
    ctx.q_to_centroids.assign(num_clusters_, 0.0F);
    for (size_t cluster_id = 0; cluster_id < num_clusters_; ++cluster_id) {
        const float* centroid = centroids_.data() + cluster_id * padded_dim_;
        const float l2 =
            rabitqlib::euclidean_sqr(ctx.rotated_query.data(), centroid, padded_dim_);
        ctx.q_to_centroids[cluster_id] = std::sqrt(l2);
    }
    if (timing) {
        timing->q_to_centroids_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - q2c_start
        ).count();
    }

    return true;
}

bool RabitQSideIndex::build_query_wrapper(QueryContext& ctx) const {
    if (!enabled_ || ctx.rotated_query.size() != padded_dim_) {
        return false;
    }
    ctx.query_wrapper = std::make_unique<rabitqlib::SplitSingleQuery<float>>(
        ctx.rotated_query.data(),
        padded_dim_,
        ex_bits_,
        query_config_,
        rabitqlib::METRIC_L2
    );
    return static_cast<bool>(ctx.query_wrapper);
}

bool RabitQSideIndex::save_query_context_cache(
    const std::string& filename,
    const std::vector<std::unique_ptr<QueryContext>>& cache
) const {
    if (!enabled_) {
        return false;
    }

    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        return false;
    }

    const uint64_t num_queries = static_cast<uint64_t>(cache.size());
    if (!write_binary(out, kRabitQQueryCtxMagic) ||
        !write_binary(out, kRabitQQueryCtxVersion) ||
        !write_binary(out, num_queries) ||
        !write_binary(out, static_cast<uint64_t>(padded_dim_)) ||
        !write_binary(out, static_cast<uint64_t>(num_clusters_)) ||
        !write_binary(out, static_cast<uint64_t>(total_bits_))) {
        return false;
    }

    for (size_t i = 0; i < cache.size(); ++i) {
        const QueryContext* ctx = cache[i].get();
        const uint8_t valid = (ctx != nullptr) ? 1U : 0U;
        if (!write_binary(out, valid)) {
            return false;
        }
        if (!ctx) {
            continue;
        }
        if (ctx->rotated_query.size() != padded_dim_ ||
            ctx->q_to_centroids.size() != num_clusters_) {
            return false;
        }
        out.write(reinterpret_cast<const char*>(ctx->rotated_query.data()),
                  static_cast<std::streamsize>(ctx->rotated_query.size() * sizeof(float)));
        out.write(reinterpret_cast<const char*>(ctx->q_to_centroids.data()),
                  static_cast<std::streamsize>(ctx->q_to_centroids.size() * sizeof(float)));
        if (!out.good()) {
            return false;
        }
    }
    return out.good();
}

bool RabitQSideIndex::load_query_context_cache(
    const std::string& filename,
    std::vector<std::unique_ptr<QueryContext>>& cache
) const {
    if (!enabled_) {
        return false;
    }

    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        return false;
    }

    uint32_t magic = 0;
    uint32_t version = 0;
    uint64_t num_queries = 0;
    uint64_t file_padded_dim = 0;
    uint64_t file_num_clusters = 0;
    uint64_t file_total_bits = 0;
    if (!read_binary(in, magic) ||
        !read_binary(in, version) ||
        !read_binary(in, num_queries) ||
        !read_binary(in, file_padded_dim) ||
        !read_binary(in, file_num_clusters) ||
        !read_binary(in, file_total_bits)) {
        return false;
    }

    if (magic != kRabitQQueryCtxMagic || version != kRabitQQueryCtxVersion) {
        return false;
    }
    if (file_padded_dim != static_cast<uint64_t>(padded_dim_) ||
        file_num_clusters != static_cast<uint64_t>(num_clusters_) ||
        file_total_bits != static_cast<uint64_t>(total_bits_)) {
        return false;
    }

    std::vector<std::unique_ptr<QueryContext>> loaded;
    loaded.resize(static_cast<size_t>(num_queries));
    for (size_t i = 0; i < loaded.size(); ++i) {
        uint8_t valid = 0U;
        if (!read_binary(in, valid)) {
            return false;
        }
        if (valid == 0U) {
            continue;
        }

        auto ctx = std::make_unique<QueryContext>();
        ctx->rotated_query.resize(padded_dim_);
        ctx->q_to_centroids.resize(num_clusters_);
        in.read(reinterpret_cast<char*>(ctx->rotated_query.data()),
                static_cast<std::streamsize>(ctx->rotated_query.size() * sizeof(float)));
        in.read(reinterpret_cast<char*>(ctx->q_to_centroids.data()),
                static_cast<std::streamsize>(ctx->q_to_centroids.size() * sizeof(float)));
        if (!in.good()) {
            return false;
        }
        if (!build_query_wrapper(*ctx)) {
            return false;
        }
        loaded[i] = std::move(ctx);
    }

    cache = std::move(loaded);
    return true;
}

float RabitQSideIndex::estimate_bin(
    ANNS::IdxType point_id,
    QueryContext& ctx,
    float* low_dist
) const {
    if (!enabled_ || point_id >= num_points_ || !ctx.query_wrapper) {
        if (low_dist) {
            *low_dist = std::numeric_limits<float>::max();
        }
        return std::numeric_limits<float>::max();
    }

    const size_t cluster_id = static_cast<size_t>(cluster_ids_[point_id]);
    if (cluster_id >= num_clusters_) {
        if (low_dist) {
            *low_dist = std::numeric_limits<float>::max();
        }
        return std::numeric_limits<float>::max();
    }

    float ip_x0_qr = 0.0F;
    float est_dist = 0.0F;
    float local_low = 0.0F;
    const float norm = ctx.q_to_centroids[cluster_id];
    const char* bin_ptr =
        reinterpret_cast<const char*>(bin_data_.data() + point_id * size_bin_data_);

    rabitqlib::split_single_estdist(
        bin_ptr,
        *ctx.query_wrapper,
        padded_dim_,
        ip_x0_qr,
        est_dist,
        local_low,
        norm * norm,
        norm
    );

    if (low_dist) {
        *low_dist = local_low;
    }
    return est_dist;
}

float RabitQSideIndex::estimate_full(
    ANNS::IdxType point_id,
    QueryContext& ctx,
    float* low_dist
) const {
    if (ex_bits_ == 0) {
        return estimate_bin(point_id, ctx, low_dist);
    }

    if (!enabled_ || point_id >= num_points_ || !ctx.query_wrapper) {
        if (low_dist) {
            *low_dist = std::numeric_limits<float>::max();
        }
        return std::numeric_limits<float>::max();
    }

    const size_t cluster_id = static_cast<size_t>(cluster_ids_[point_id]);
    if (cluster_id >= num_clusters_) {
        if (low_dist) {
            *low_dist = std::numeric_limits<float>::max();
        }
        return std::numeric_limits<float>::max();
    }

    float est_dist = 0.0F;
    float local_low = 0.0F;
    float ip_x0_qr = 0.0F;
    const float norm = ctx.q_to_centroids[cluster_id];
    const char* bin_ptr =
        reinterpret_cast<const char*>(bin_data_.data() + point_id * size_bin_data_);
    const char* ex_ptr =
        reinterpret_cast<const char*>(ex_data_.data() + point_id * size_ex_data_);

    rabitqlib::split_single_fulldist(
        bin_ptr,
        ex_ptr,
        ip_func_,
        *ctx.query_wrapper,
        padded_dim_,
        ex_bits_,
        est_dist,
        local_low,
        ip_x0_qr,
        norm * norm,
        norm
    );

    if (low_dist) {
        *low_dist = local_low;
    }
    return est_dist;
}

}  // namespace ANNS::rabitq
