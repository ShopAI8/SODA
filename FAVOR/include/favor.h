#pragma once

#include <cstring>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <iostream>
#include <mutex>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #ifdef USE_SSE
        #include <immintrin.h>
    #endif
#endif

#include "hnswlib/hnswlib.h"
#include "hnswlib/hnswalg.h"
#include "filter_condition.h"
#include "check.h"

using namespace hnswlib;
namespace favor
{

    template <typename dist_t>
    class FAVOR : public HierarchicalNSW<dist_t>
    {
    public:
        using HierarchicalNSW<dist_t>::searchKnn;
        using HierarchicalNSW<dist_t>::addPoint;
        static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
        size_t num_attribute_{0};
        size_t attribute_offset_{0};
        std::mutex delta_mutex;
        dist_t delta_d = 0.0;
        const dist_t LARGE_DIST = 100000.0;

    private:
    public:

        FAVOR(
            SpaceInterface<dist_t> *s,
            const std::string &location,
            bool nmslib = false,
            size_t max_elements = 0,
            size_t ef = 100,
            bool allow_replace_deleted = false)
            : HierarchicalNSW<dist_t>(max_elements, allow_replace_deleted)
        {
            loadIndex(location, s, max_elements, ef);
        }

        FAVOR(
            SpaceInterface<dist_t> *s,
            size_t max_elements,
            size_t M = 16,
            size_t ef_construction = 200,
            size_t num_attribute = 1,
            size_t random_seed = 100,
            bool allow_replace_deleted = false) : HierarchicalNSW<dist_t>(max_elements, allow_replace_deleted)
        {
            this->max_elements_ = max_elements;
            this->num_deleted_ = 0;
            this->data_size_ = s->get_data_size();
            this->fstdistfunc_ = s->get_dist_func();
            this->dist_func_param_ = s->get_dist_func_param();
            if (M <= 10000)
            {
                this->M_ = M;
            }
            else
            {
                HNSWERR << "warning: M parameter exceeds 10000 which may lead to adverse effects." << std::endl;
                HNSWERR << "         Cap to 10000 will be applied for the rest of the processing." << std::endl;
                this->M_ = 10000;
            }
            this->maxM_ = this->M_;
            this->maxM0_ = this->M_ * 2;
            this->ef_construction_ = std::max(ef_construction, this->M_);
            this->ef_ = 100;
            num_attribute_ = num_attribute;

            this->level_generator_.seed(random_seed);
            this->update_probability_generator_.seed(random_seed + 1);

            this->size_links_level0_ = this->maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
            this->size_data_per_element_ = this->size_links_level0_ + this->data_size_ + sizeof(labeltype) + sizeof(attributetype) * num_attribute_;
            this->offsetData_ = this->size_links_level0_;
            this->label_offset_ = this->size_links_level0_ + this->data_size_;
            attribute_offset_ = this->size_links_level0_ + this->data_size_ + sizeof(labeltype);
            this->offsetLevel0_ = 0;

            this->data_level0_memory_ = static_cast<char*>(malloc(this->max_elements_ * this->size_data_per_element_));
            if (this->data_level0_memory_ == nullptr)
                throw std::runtime_error("Not enough memory");

            this->cur_element_count = 0;

            this->visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));

            this->enterpoint_node_ = -1;
            this->maxlevel_ = -1;

            this->linkLists_ = static_cast<char**>(malloc(sizeof(void *) * this->max_elements_));
            if (this->linkLists_ == nullptr)
                throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
            this->size_links_per_element_ = this->maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
            this->mult_ = 1.0f / std::log(1.0f * static_cast<float>(this->M_));
            this->revSize_ = 1.0f / this->mult_;
        }

        struct CompareByFirst_
        {
            constexpr bool operator()(std::pair<dist_t, tableint> const &a,
                                      std::pair<dist_t, tableint> const &b) const noexcept
            {
                return a.first < b.first;
            }
        };

        dist_t distFilter(float p) const
        {
            return (1.0f - p) * (static_cast<float>(this->ef_) - p) * delta_d / (2.0f * p);
        }

        inline void setAttribute(tableint internal_id, attributetype *attribute) const
        {
            if (this->num_attribute_ == 0) {
                return;
            }
            memcpy((this->data_level0_memory_ + internal_id * this->size_data_per_element_ + attribute_offset_),
                   attribute, this->num_attribute_ * sizeof(attributetype));
        }

        inline attributetype *getAttributeByInternalId(tableint internal_id) const
        {
            return reinterpret_cast<attributetype*>(this->data_level0_memory_ + 
                                                     internal_id * this->size_data_per_element_ + attribute_offset_);
        }

        void saveIndex(const std::string &location)
        {
            std::ofstream output(location, std::ios::binary);
            std::streampos position;

            writeBinaryPOD(output, this->offsetLevel0_);
            writeBinaryPOD(output, this->max_elements_);
            writeBinaryPOD(output, this->cur_element_count);
            writeBinaryPOD(output, this->size_data_per_element_);
            writeBinaryPOD(output, this->label_offset_);
            writeBinaryPOD(output, this->offsetData_);
            writeBinaryPOD(output, this->maxlevel_);
            writeBinaryPOD(output, this->enterpoint_node_);
            writeBinaryPOD(output, this->maxM_);

            writeBinaryPOD(output, this->maxM0_);
            writeBinaryPOD(output, this->M_);
            writeBinaryPOD(output, this->mult_);
            writeBinaryPOD(output, this->ef_construction_);

            writeBinaryPOD(output, num_attribute_);
            writeBinaryPOD(output, attribute_offset_);
            writeBinaryPOD(output, delta_d);

            output.write(this->data_level0_memory_, 
                         static_cast<std::streamsize>(this->cur_element_count * this->size_data_per_element_));

            for (size_t i = 0; i < this->cur_element_count; i++)
            {
                unsigned int linkListSize = this->element_levels_[i] > 0 ? 
                    static_cast<unsigned int>(this->size_links_per_element_ * this->element_levels_[i]) : 0;
                writeBinaryPOD(output, linkListSize);
                if (linkListSize)
                    output.write(this->linkLists_[i], static_cast<std::streamsize>(linkListSize));
            }
            output.close();
        }

        void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, 
                       size_t max_elements_i = 0, size_t ef = 100)
        {
            std::ifstream input(location, std::ios::binary);

            if (!input.is_open())
                throw std::runtime_error("Cannot open file");

            this->clear();
            input.seekg(0, input.end);
            std::streampos total_filesize = input.tellg();
            input.seekg(0, input.beg);

            readBinaryPOD(input, this->offsetLevel0_);
            readBinaryPOD(input, this->max_elements_);
            readBinaryPOD(input, this->cur_element_count);

            size_t max_elements = max_elements_i;
            if (max_elements < this->cur_element_count)
                max_elements = this->max_elements_;
            this->max_elements_ = max_elements;
            readBinaryPOD(input, this->size_data_per_element_);
            readBinaryPOD(input, this->label_offset_);
            readBinaryPOD(input, this->offsetData_);
            readBinaryPOD(input, this->maxlevel_);
            readBinaryPOD(input, this->enterpoint_node_);

            readBinaryPOD(input, this->maxM_);
            readBinaryPOD(input, this->maxM0_);
            readBinaryPOD(input, this->M_);
            readBinaryPOD(input, this->mult_);
            readBinaryPOD(input, this->ef_construction_);

            readBinaryPOD(input, num_attribute_);
            readBinaryPOD(input, attribute_offset_);
            readBinaryPOD(input, delta_d);

            this->data_size_ = s->get_data_size();
            this->fstdistfunc_ = s->get_dist_func();
            this->dist_func_param_ = s->get_dist_func_param();

            auto pos = input.tellg();

            input.seekg(static_cast<std::streamoff>(this->cur_element_count * this->size_data_per_element_), input.cur);
            for (size_t i = 0; i < this->cur_element_count; i++)
            {
                if (input.tellg() < 0 || input.tellg() >= total_filesize)
                {
                    throw std::runtime_error("Index seems to be corrupted or unsupported");
                }

                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize != 0)
                {
                    input.seekg(static_cast<std::streamoff>(linkListSize), input.cur);
                }
            }

            if (input.tellg() != total_filesize)
                throw std::runtime_error("Index seems to be corrupted or unsupported");

            input.clear();
            input.seekg(pos, input.beg);

            this->data_level0_memory_ = static_cast<char*>(malloc(max_elements * this->size_data_per_element_));
            if (this->data_level0_memory_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
            input.read(this->data_level0_memory_, 
                       static_cast<std::streamsize>(this->cur_element_count * this->size_data_per_element_));

            this->size_links_per_element_ = this->maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
            this->size_links_level0_ = this->maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
            std::vector<std::mutex>(max_elements).swap(this->link_list_locks_);
            std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(this->label_op_locks_);

            this->visited_list_pool_.reset(new VisitedListPool(1, max_elements));

            this->linkLists_ = static_cast<char**>(malloc(sizeof(void *) * max_elements));
            if (this->linkLists_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
            this->element_levels_ = std::vector<int>(max_elements);
            this->revSize_ = 1.0f / this->mult_;
            this->ef_ = ef;
            for (size_t i = 0; i < this->cur_element_count; i++)
            {
                this->label_lookup_[this->getExternalLabel(i)] = i;
                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize == 0)
                {
                    this->element_levels_[i] = 0;
                    this->linkLists_[i] = nullptr;
                }
                else
                {
                    this->element_levels_[i] = static_cast<int>(linkListSize / this->size_links_per_element_);
                    this->linkLists_[i] = static_cast<char*>(malloc(linkListSize));
                    if (this->linkLists_[i] == nullptr)
                        throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                    input.read(this->linkLists_[i], static_cast<std::streamsize>(linkListSize));
                }
            }

            input.close();
            return;
        }

        void getNeighborsByHeuristic2(
            std::priority_queue<std::pair<dist_t, tableint>, 
                                std::vector<std::pair<dist_t, tableint>>, 
                                CompareByFirst_> &top_candidates,
            const size_t M)
        {
            if (top_candidates.size() < M)
            {
                return;
            }

            std::vector<std::pair<dist_t, tableint>> candidates;
            candidates.reserve(top_candidates.size());
            while (!top_candidates.empty())
            {
                candidates.push_back(top_candidates.top());
                top_candidates.pop();
            }

            std::reverse(candidates.begin(), candidates.end());
            std::vector<std::pair<dist_t, tableint>> return_list;
            return_list.reserve(M);

            for (const auto &curent_pair : candidates)
            {
                if (return_list.size() >= M)
                    break;
                
                bool good = true;
                dist_t dist_to_query = curent_pair.first;

                for (const auto &second_pair : return_list)
                {
                    dist_t curdist =
                        this->fstdistfunc_(this->getDataByInternalId(second_pair.second),
                                           this->getDataByInternalId(curent_pair.second),
                                           this->dist_func_param_);
                    if (curdist < dist_to_query)
                    {
                        good = false;
                        break;
                    }
                }
                if (good)
                {
                    return_list.push_back(curent_pair);
                }
            }

            for (const auto &curent_pair : return_list)
            {
                top_candidates.emplace(curent_pair.first, curent_pair.second);
            }
        }

        tableint mutuallyConnectNewElement(
            const void *data_point,
            tableint cur_c,
            std::priority_queue<std::pair<dist_t, tableint>, 
                                std::vector<std::pair<dist_t, tableint>>, 
                                CompareByFirst_> &top_candidates,
            int level)
        {
            size_t Mcurmax = level ? this->maxM_ : this->maxM0_;
            this->getNeighborsByHeuristic2(top_candidates, this->M_);
            if (top_candidates.size() > this->M_)
                throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

            std::vector<tableint> selectedNeighbors;
            selectedNeighbors.reserve(this->M_);
            while (top_candidates.size() > 0)
            {
                selectedNeighbors.push_back(top_candidates.top().second);
                top_candidates.pop();
            }
            tableint next_closest_entry_point = selectedNeighbors.back();

            {
                std::unique_lock<std::mutex> lock(this->link_list_locks_[cur_c], std::defer_lock);
                linklistsizeint *ll_cur;
                if (level == 0)
                    ll_cur = this->get_linklist0(cur_c);
                else
                    ll_cur = this->get_linklist(cur_c, level);

                if (*ll_cur)
                {
                    throw std::runtime_error("The newly inserted element should have blank link list");
                }
                this->setListCount(ll_cur, selectedNeighbors.size());
                tableint *data = reinterpret_cast<tableint*>(ll_cur + 1);
                for (size_t idx = 0; idx < selectedNeighbors.size(); idx++)
                {
                    if (data[idx])
                        throw std::runtime_error("Possible memory corruption");
                    if (level > this->element_levels_[selectedNeighbors[idx]])
                        throw std::runtime_error("Trying to make a link on a non-existent level");

                    data[idx] = selectedNeighbors[idx];
                }
            }
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++)
            {
                std::unique_lock<std::mutex> lock(this->link_list_locks_[selectedNeighbors[idx]]);

                linklistsizeint *ll_other;
                if (level == 0)
                    ll_other = this->get_linklist0(selectedNeighbors[idx]);
                else
                    ll_other = this->get_linklist(selectedNeighbors[idx], level);

                size_t sz_link_list_other = this->getListCount(ll_other);

                if (sz_link_list_other > Mcurmax)
                    throw std::runtime_error("Bad value of sz_link_list_other");
                if (selectedNeighbors[idx] == cur_c)
                    throw std::runtime_error("Trying to connect an element to itself");
                if (level > this->element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                tableint *data = reinterpret_cast<tableint*>(ll_other + 1);
                bool is_cur_c_present = false;

                if (!is_cur_c_present)
                {
                    if (sz_link_list_other < Mcurmax)
                    {
                        data[sz_link_list_other] = cur_c;
                        this->setListCount(ll_other, sz_link_list_other + 1);
                    }
                    else
                    {
                        dist_t d_max = this->fstdistfunc_(this->getDataByInternalId(cur_c), 
                                                           this->getDataByInternalId(selectedNeighbors[idx]),
                                                           this->dist_func_param_);
                        std::priority_queue<std::pair<dist_t, tableint>, 
                                           std::vector<std::pair<dist_t, tableint>>, 
                                           CompareByFirst_> candidates;
                        candidates.emplace(d_max, cur_c);

                        for (size_t j = 0; j < sz_link_list_other; j++)
                        {
                            candidates.emplace(
                                this->fstdistfunc_(this->getDataByInternalId(data[j]), 
                                                  this->getDataByInternalId(selectedNeighbors[idx]),
                                                  this->dist_func_param_),
                                data[j]);
                        }

                        this->getNeighborsByHeuristic2(candidates, Mcurmax);

                        int indx = 0;
                        while (candidates.size() > 0)
                        {
                            data[indx] = candidates.top().second;
                            candidates.pop();
                            indx++;
                        }

                        this->setListCount(ll_other, static_cast<linklistsizeint>(indx));
                    }
                }
            }

            return next_closest_entry_point;
        }

        std::priority_queue<std::pair<dist_t, tableint>, 
                           std::vector<std::pair<dist_t, tableint>>, 
                           CompareByFirst_>
        searchBaseLayerFilter(tableint ep_id, const void *data_point, int layer)
        {
            VisitedList *vl = this->visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, 
                               std::vector<std::pair<dist_t, tableint>>, 
                               CompareByFirst_> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, 
                               std::vector<std::pair<dist_t, tableint>>, 
                               CompareByFirst_> candidateSet;

            dist_t lowerBound;
            if (!this->isMarkedDeleted(ep_id))
            {
                dist_t dist = this->fstdistfunc_(data_point, this->getDataByInternalId(ep_id), this->dist_func_param_);
                top_candidates.emplace(dist, ep_id);
                lowerBound = dist;
                candidateSet.emplace(-dist, ep_id);
            }
            else
            {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidateSet.emplace(-lowerBound, ep_id);
            }
            visited_array[ep_id] = visited_array_tag;

            while (!candidateSet.empty())
            {
                std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
                if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == this->ef_construction_)
                {
                    break;
                }
                candidateSet.pop();

                tableint curNodeNum = curr_el_pair.second;
                std::unique_lock<std::mutex> lock(this->link_list_locks_[curNodeNum]);

                int *data;
                if (layer == 0)
                    data = reinterpret_cast<int*>(this->get_linklist0(curNodeNum));
                else
                    data = reinterpret_cast<int*>(this->get_linklist(curNodeNum, layer));

                size_t size = this->getListCount(reinterpret_cast<linklistsizeint*>(data));
                tableint *datal = reinterpret_cast<tableint*>(data + 1);
#ifdef USE_SSE
                _mm_prefetch(reinterpret_cast<char*>(visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<char*>(visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(this->getDataByInternalId(*datal), _MM_HINT_T0);
                _mm_prefetch(this->getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

                for (size_t j = 0; j < size; j++)
                {
                    tableint candidate_id = *(datal + j);
#ifdef USE_SSE
                    _mm_prefetch(reinterpret_cast<char*>(visited_array + *(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(this->getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                    if (visited_array[candidate_id] == visited_array_tag)
                        continue;
                    visited_array[candidate_id] = visited_array_tag;
                    char *currObj1 = (this->getDataByInternalId(candidate_id));

                    dist_t dist1 = this->fstdistfunc_(data_point, currObj1, this->dist_func_param_);
                    if (top_candidates.size() < this->ef_construction_ || lowerBound > dist1)
                    {
                        candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(this->getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                        if (!this->isMarkedDeleted(candidate_id))
                            top_candidates.emplace(dist1, candidate_id);

                        if (top_candidates.size() > this->ef_construction_)
                            top_candidates.pop();

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
            this->visited_list_pool_->releaseVisitedList(vl);
            return top_candidates;
        }

        std::priority_queue<std::pair<dist_t, tableint>, 
                           std::vector<std::pair<dist_t, tableint>>, 
                           CompareByFirst_>
        searchBaseLayerSTFilter(tableint ep_id,
                                const void *data_point,
                                size_t ef,
                                size_t k,
                                dist_t e_distance,
                                const OptimizedFilter& conditions) const
        {
            VisitedList *vl = this->visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>, 
                               std::vector<std::pair<dist_t, tableint>>, 
                               CompareByFirst_> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, 
                               std::vector<std::pair<dist_t, tableint>>, 
                               CompareByFirst_> candidate_set;

            dist_t lowerBound;
            char *ep_data = this->getDataByInternalId(ep_id);
            dist_t dist;
            
            bool ep_qualified = conditions.check(getAttributeByInternalId(ep_id));
            
            dist_t base_dist = this->fstdistfunc_(data_point, ep_data, this->dist_func_param_);
            dist = ep_qualified ? base_dist : base_dist + e_distance;
            
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            candidate_set.emplace(-dist, ep_id);

            visited_array[ep_id] = visited_array_tag;
            size_t num_in_range = ep_qualified ? 1 : 0;

            while (!candidate_set.empty())
            {
                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
                dist_t candidate_dist = -current_node_pair.first;

                if (candidate_dist > 0.95 * lowerBound && num_in_range > k * 0.5)
                    break;

                candidate_set.pop();
                tableint current_node_id = current_node_pair.second;
                int *data = reinterpret_cast<int*>(this->get_linklist0(current_node_id));
                size_t size = this->getListCount(reinterpret_cast<linklistsizeint*>(data));

#ifdef USE_SSE
                _mm_prefetch(reinterpret_cast<char*>(visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<char*>(visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(this->data_level0_memory_ + 
                            static_cast<size_t>(*(data + 1)) * this->size_data_per_element_ + this->offsetData_, 
                            _MM_HINT_T0);
                if (size > 1) {
                    _mm_prefetch(reinterpret_cast<char*>(data + 2), _MM_HINT_T0);
                }
#endif

                size_t j = 1;
                for (; j + 1 <= size; j += 2)
                {
                    int candidate_id1 = *(data + j);
                    int candidate_id2 = *(data + j + 1);
                    
#ifdef USE_SSE
                    if (j + 2 <= size) {
                        _mm_prefetch(reinterpret_cast<char*>(visited_array + *(data + j + 2)), _MM_HINT_T0);
                        _mm_prefetch(this->data_level0_memory_ + 
                                    static_cast<size_t>(*(data + j + 2)) * this->size_data_per_element_ + this->offsetData_,
                                    _MM_HINT_T0);
                    }
#endif
                    if (visited_array[candidate_id1] != visited_array_tag)
                    {
                        visited_array[candidate_id1] = visited_array_tag;
                        char *currObj1 = this->getDataByInternalId(candidate_id1);
                        
                        bool candidate_qualified = conditions.check(getAttributeByInternalId(candidate_id1));
                        // Note: Exact distance computation for NTD vectors is unnecessary here.
                        dist_t dist1 = candidate_qualified ? 
                            this->fstdistfunc_(data_point, currObj1, this->dist_func_param_) :
                            FastApproxL2Sqr(data_point, currObj1, this->dist_func_param_) + e_distance;
                        
                        if (top_candidates.size() < ef || lowerBound > dist1)
                        {
                            candidate_set.emplace(-dist1, candidate_id1);
                            if (candidate_qualified)
                                num_in_range++;
                            top_candidates.emplace(dist1, candidate_id1);

                            if (top_candidates.size() > ef)
                            {
                                auto evicted = top_candidates.top();
                                top_candidates.pop();
                                
                                if (conditions.check(getAttributeByInternalId(evicted.second)))
                                    num_in_range--;
                            }

                            if (!top_candidates.empty())
                                lowerBound = top_candidates.top().first;
                        }
                    }
                    
                    if (visited_array[candidate_id2] != visited_array_tag)
                    {
                        visited_array[candidate_id2] = visited_array_tag;
                        char *currObj2 = this->getDataByInternalId(candidate_id2);
                        
                        bool candidate_qualified = conditions.check(getAttributeByInternalId(candidate_id2));
                        dist_t dist2 = candidate_qualified ? 
                            this->fstdistfunc_(data_point, currObj2, this->dist_func_param_) :
                            FastApproxL2Sqr(data_point, currObj2, this->dist_func_param_) + e_distance;
                        
                        if (top_candidates.size() < ef || lowerBound > dist2)
                        {
                            candidate_set.emplace(-dist2, candidate_id2);
                            if (candidate_qualified)
                                num_in_range++;
                            top_candidates.emplace(dist2, candidate_id2);

                            if (top_candidates.size() > ef)
                            {
                                auto evicted = top_candidates.top();
                                top_candidates.pop();
                                
                                if (conditions.check(getAttributeByInternalId(evicted.second)))
                                    num_in_range--;
                            }

                            if (!top_candidates.empty())
                                lowerBound = top_candidates.top().first;
                        }
                    }
                }
                
                for (; j <= size; ++j)
                {
                    int candidate_id = *(data + j);
                    if (visited_array[candidate_id] != visited_array_tag)
                    {
                        visited_array[candidate_id] = visited_array_tag;
                        char *currObj1 = this->getDataByInternalId(candidate_id);
                        
                        bool candidate_qualified = conditions.check(getAttributeByInternalId(candidate_id));
                        dist_t dist1 = candidate_qualified ? 
                            this->fstdistfunc_(data_point, currObj1, this->dist_func_param_) :
                            FastApproxL2Sqr(data_point, currObj1, this->dist_func_param_) + e_distance;
                        
                        if (top_candidates.size() < ef || lowerBound > dist1)
                        {
                            candidate_set.emplace(-dist1, candidate_id);
                            if (candidate_qualified)
                                num_in_range++;
                            top_candidates.emplace(dist1, candidate_id);

                            if (top_candidates.size() > ef)
                            {
                                auto evicted = top_candidates.top();
                                top_candidates.pop();
                                
                                if (conditions.check(getAttributeByInternalId(evicted.second)))
                                    num_in_range--;
                            }

                            if (!top_candidates.empty())
                                lowerBound = top_candidates.top().first;
                        }
                    }
                }
            }

            this->visited_list_pool_->releaseVisitedList(vl);
            return top_candidates;
        }

        void addPoint(const void *data_point, labeltype label, attributetype *attribute, bool replace_deleted = false)
        {
            if ((this->allow_replace_deleted_ == false) && (replace_deleted == true))
                throw std::runtime_error("Replacement of deleted elements is disabled in constructor");

            std::unique_lock<std::mutex> lock_label(this->getLabelOpMutex(label));
            if (!replace_deleted)
            {
                addPoint(data_point, label, -1, attribute);
                return;
            }
            tableint internal_id_replaced;
            std::unique_lock<std::mutex> lock_deleted_elements(this->deleted_elements_lock);
            bool is_vacant_place = !this->deleted_elements.empty();
            if (is_vacant_place)
            {
                internal_id_replaced = *this->deleted_elements.begin();
                this->deleted_elements.erase(internal_id_replaced);
            }
            lock_deleted_elements.unlock();

            if (!is_vacant_place)
            {
                addPoint(data_point, label, -1, attribute);
            }
            else
            {
                labeltype label_replaced = this->getExternalLabel(internal_id_replaced);
                this->setExternalLabel(internal_id_replaced, label);
                setAttribute(internal_id_replaced, attribute);

                std::unique_lock<std::mutex> lock_table(this->label_lookup_lock);
                this->label_lookup_.erase(label_replaced);
                this->label_lookup_[label] = internal_id_replaced;
                lock_table.unlock();

                this->unmarkDeletedInternal(internal_id_replaced);
                this->updatePoint(data_point, internal_id_replaced, 1.0);
            }
        }

        tableint addPoint(const void *data_point, labeltype label, int level, attributetype *attribute)
        {
            tableint cur_c = 0;
            {
                std::unique_lock<std::mutex> lock_table(this->label_lookup_lock);
                auto search = this->label_lookup_.find(label);
                if (search != this->label_lookup_.end())
                {
                    tableint existingInternalId = search->second;
                    if (this->allow_replace_deleted_)
                    {
                        if (this->isMarkedDeleted(existingInternalId))
                            throw std::runtime_error("Can't use addPoint to update deleted elements if replacement of deleted elements is enabled.");
                    }
                    lock_table.unlock();

                    if (this->isMarkedDeleted(existingInternalId))
                        this->unmarkDeletedInternal(existingInternalId);
                    this->updatePoint(data_point, existingInternalId, 1.0);
                    return existingInternalId;
                }

                if (this->cur_element_count >= this->max_elements_)
                    throw std::runtime_error("The number of elements exceeds the specified limit");

                cur_c = this->cur_element_count;
                this->cur_element_count++;
                this->label_lookup_[label] = cur_c;
            }

            std::unique_lock<std::mutex> lock_el(this->link_list_locks_[cur_c]);
            int curlevel = this->getRandomLevel(this->mult_);
            if (level > 0)
                curlevel = level;

            this->element_levels_[cur_c] = curlevel;

            std::unique_lock<std::mutex> templock(this->global);
            int maxlevelcopy = this->maxlevel_;
            if (curlevel <= maxlevelcopy)
                templock.unlock();
            tableint currObj = this->enterpoint_node_;

            memset(this->data_level0_memory_ + cur_c * this->size_data_per_element_ + this->offsetLevel0_, 
                   0, this->size_data_per_element_);

            memcpy(this->getExternalLabeLp(cur_c), &label, sizeof(labeltype));
            memcpy(this->getDataByInternalId(cur_c), data_point, this->data_size_);
            setAttribute(cur_c, attribute);

            if (curlevel)
            {
                this->linkLists_[cur_c] = static_cast<char*>(malloc(this->size_links_per_element_ * curlevel + 1));
                if (this->linkLists_[cur_c] == nullptr)
                    throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
                memset(this->linkLists_[cur_c], 0, this->size_links_per_element_ * curlevel + 1);
            }

            dist_t local_delta_d = 0;

            if (static_cast<signed>(currObj) != -1)
            {
                if (curlevel < maxlevelcopy)
                {
                    dist_t curdist = this->fstdistfunc_(data_point, this->getDataByInternalId(currObj), this->dist_func_param_);
                    for (int lvl = maxlevelcopy; lvl > curlevel; lvl--)
                    {
                        bool changed = true;
                        while (changed)
                        {
                            changed = false;
                            unsigned int *data;
                            std::unique_lock<std::mutex> lock(this->link_list_locks_[currObj]);
                            data = this->get_linklist(currObj, lvl);
                            int size = this->getListCount(data);

                            tableint *datal = reinterpret_cast<tableint*>(data + 1);
                            for (int i = 0; i < size; i++)
                            {
                                tableint cand = datal[i];
                                if (cand < 0 || cand > this->max_elements_)
                                    throw std::runtime_error("cand error");
                                dist_t d = this->fstdistfunc_(data_point, this->getDataByInternalId(cand), this->dist_func_param_);
                                if (d < curdist)
                                {
                                    curdist = d;
                                    currObj = cand;
                                    changed = true;
                                }
                            }
                        }
                    }
                }

                for (int lvl = std::min(curlevel, maxlevelcopy); lvl >= 0; lvl--)
                {
                    if (lvl > maxlevelcopy || lvl < 0)
                        throw std::runtime_error("Level error");
                    std::priority_queue<std::pair<dist_t, tableint>, 
                                       std::vector<std::pair<dist_t, tableint>>, 
                                       CompareByFirst_> top_candidates = searchBaseLayerFilter(
                        currObj, data_point, lvl);

                    if (lvl == 0 && top_candidates.size() == this->ef_construction_)
                    {
                        dist_t rate = static_cast<dist_t>(this->ef_construction_ - 10);
                        auto temp = top_candidates;
                        std::vector<std::pair<dist_t, tableint>> elements;
                        while (!temp.empty())
                        {
                            elements.push_back(temp.top());
                            temp.pop();
                        }

                        dist_t diff = elements[0].first - elements[elements.size() - 10].first;
                        local_delta_d += 5.0f * diff / (rate * static_cast<dist_t>(this->max_elements_));
                    }
                    currObj = this->mutuallyConnectNewElement(data_point, cur_c, top_candidates, lvl);
                }
            }
            else
            {
                this->enterpoint_node_ = 0;
                this->maxlevel_ = curlevel;
            }

            std::lock_guard<std::mutex> lock(delta_mutex);
            delta_d += local_delta_d;

            if (curlevel > maxlevelcopy)
            {
                this->enterpoint_node_ = cur_c;
                this->maxlevel_ = curlevel;
            }
            return cur_c;
        }

        std::priority_queue<std::pair<dist_t, labeltype>>
        searchGraph(const void *query_data, size_t k, float p, const OptimizedFilter& conditions) const
        {
            std::priority_queue<std::pair<dist_t, labeltype>> result;
            if (this->cur_element_count == 0)
                return result;

            dist_t e_distance = distFilter(p);

            tableint currObj = this->enterpoint_node_;
            dist_t curdist = this->fstdistfunc_(query_data, this->getDataByInternalId(this->enterpoint_node_), this->dist_func_param_);
            for (int level = this->maxlevel_; level > 0; level--)
            {
                bool changed = true;
                while (changed)
                {
                    changed = false;
                    unsigned int *data = reinterpret_cast<unsigned int*>(this->get_linklist(currObj, level));
                    int size = this->getListCount(data);

                    tableint *datal = reinterpret_cast<tableint*>(data + 1);
                    for (int i = 0; i < size; i++)
                    {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > this->max_elements_)
                            throw std::runtime_error("cand error");
                        dist_t d = this->fstdistfunc_(query_data, this->getDataByInternalId(cand), this->dist_func_param_);
                        if (d < curdist)
                        {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }

            size_t ef = std::max(this->ef_, k);
            auto top_candidates = searchBaseLayerSTFilter(currObj, query_data, ef, k, e_distance / static_cast<dist_t>(this->ef_), conditions);
            
            std::vector<std::pair<dist_t, labeltype>> valid_results;
            valid_results.reserve(std::min(k, top_candidates.size()));
            
            while (!top_candidates.empty())
            {
                std::pair<dist_t, tableint> rez = top_candidates.top();
                
                attributetype *attribute = getAttributeByInternalId(rez.second);
                if (conditions.check(attribute))
                {
                    valid_results.emplace_back(rez.first, this->getExternalLabel(rez.second));
                }
                top_candidates.pop();
            }
            
            for (const auto &valid_result : valid_results)
                result.push(valid_result);
            
            while (result.size() < k)
                result.push(std::pair<dist_t, labeltype>(1000000.0f, -1));
            
            while (result.size() > k)
                result.pop();

            return result;
        }

        template <typename GenericFilter>
        std::priority_queue<std::pair<dist_t, labeltype>>
        searchGraphGeneric(const void *query_data, size_t k, float p, const GenericFilter& conditions) const
        {
            std::priority_queue<std::pair<dist_t, labeltype>> result;
            if (this->cur_element_count == 0)
                return result;

            dist_t e_distance = distFilter(p);

            tableint currObj = this->enterpoint_node_;
            dist_t curdist = this->fstdistfunc_(query_data, this->getDataByInternalId(this->enterpoint_node_), this->dist_func_param_);
            for (int level = this->maxlevel_; level > 0; level--)
            {
                bool changed = true;
                while (changed)
                {
                    changed = false;
                    unsigned int *data = reinterpret_cast<unsigned int*>(this->get_linklist(currObj, level));
                    int size = this->getListCount(data);

                    tableint *datal = reinterpret_cast<tableint*>(data + 1);
                    for (int i = 0; i < size; i++)
                    {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > this->max_elements_)
                            throw std::runtime_error("cand error");
                        dist_t d = this->fstdistfunc_(query_data, this->getDataByInternalId(cand), this->dist_func_param_);
                        if (d < curdist)
                        {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }

            size_t ef = std::max(this->ef_, k);
            VisitedList *vl = this->visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            std::priority_queue<std::pair<dist_t, tableint>,
                               std::vector<std::pair<dist_t, tableint>>,
                               CompareByFirst_> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>,
                               std::vector<std::pair<dist_t, tableint>>,
                               CompareByFirst_> candidate_set;

            bool ep_qualified = conditions.check(currObj);
            char *ep_data = this->getDataByInternalId(currObj);
            dist_t base_dist = this->fstdistfunc_(query_data, ep_data, this->dist_func_param_);
            dist_t dist = ep_qualified ? base_dist : base_dist + e_distance / static_cast<dist_t>(this->ef_);
            dist_t lowerBound = dist;
            top_candidates.emplace(dist, currObj);
            candidate_set.emplace(-dist, currObj);
            visited_array[currObj] = visited_array_tag;
            size_t num_in_range = ep_qualified ? 1 : 0;

            while (!candidate_set.empty())
            {
                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
                dist_t candidate_dist = -current_node_pair.first;

                if (candidate_dist > 0.95 * lowerBound && num_in_range > k * 0.5)
                    break;

                candidate_set.pop();
                tableint current_node_id = current_node_pair.second;
                int *data = reinterpret_cast<int*>(this->get_linklist0(current_node_id));
                size_t size = this->getListCount(reinterpret_cast<linklistsizeint*>(data));

                for (size_t j = 1; j <= size; ++j)
                {
                    int candidate_id = *(data + j);
                    if (visited_array[candidate_id] == visited_array_tag)
                        continue;

                    visited_array[candidate_id] = visited_array_tag;
                    char *currObj1 = this->getDataByInternalId(candidate_id);
                    bool candidate_qualified = conditions.check(candidate_id);
                    dist_t dist1 = candidate_qualified ?
                        this->fstdistfunc_(query_data, currObj1, this->dist_func_param_) :
                        FastApproxL2Sqr(query_data, currObj1, this->dist_func_param_) + e_distance / static_cast<dist_t>(this->ef_);

                    if (top_candidates.size() < ef || lowerBound > dist1)
                    {
                        candidate_set.emplace(-dist1, candidate_id);
                        if (candidate_qualified)
                            num_in_range++;
                        top_candidates.emplace(dist1, candidate_id);

                        if (top_candidates.size() > ef)
                        {
                            auto evicted = top_candidates.top();
                            top_candidates.pop();
                            if (conditions.check(evicted.second))
                                num_in_range--;
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }

            this->visited_list_pool_->releaseVisitedList(vl);

            std::vector<std::pair<dist_t, labeltype>> valid_results;
            valid_results.reserve(std::min(k, top_candidates.size()));
            while (!top_candidates.empty())
            {
                std::pair<dist_t, tableint> rez = top_candidates.top();
                if (conditions.check(rez.second))
                {
                    valid_results.emplace_back(rez.first, this->getExternalLabel(rez.second));
                }
                top_candidates.pop();
            }

            for (const auto &valid_result : valid_results)
                result.push(valid_result);

            while (result.size() < k)
                result.push(std::pair<dist_t, labeltype>(1000000.0f, -1));

            while (result.size() > k)
                result.pop();

            return result;
        }

        std::priority_queue<std::pair<dist_t, labeltype>>
        searchBruteForce(const void *query_data, size_t k, const OptimizedFilter& conditions) const
        {
            std::priority_queue<std::pair<dist_t, labeltype>> result;
            std::vector<size_t> candidates;
            candidates.reserve(this->max_elements_ / 10);

#ifdef USE_AVX2
            conditions.checkBatchAVX2(
                this->data_level0_memory_,
                this->size_data_per_element_,
                this->attribute_offset_,
                0,
                this->max_elements_,
                candidates
            );
#else
            for (size_t i = 0; i < this->max_elements_; i++)
            {
                if (conditions.check(getAttributeByInternalId(i)))  [[unlikely]]
                {
                    candidates.push_back(i);
                }
            }
#endif
            
            for (size_t idx : candidates)
            {
                dist_t d = this->fstdistfunc_(query_data, this->getDataByInternalId(idx), this->dist_func_param_);
                
                if (result.size() < k)
                    result.push(std::pair<dist_t, labeltype>(d, this->getExternalLabel(idx)));
                else if (d < result.top().first)
                {
                    result.push(std::pair<dist_t, labeltype>(d, this->getExternalLabel(idx)));
                    result.pop();
                }
            }
            
            while (result.size() < k)
                result.push(std::pair<dist_t, labeltype>(1000000.0f, -1));
            return result;
        }

        template <typename GenericFilter>
        std::priority_queue<std::pair<dist_t, labeltype>>
        searchBruteForceGeneric(const void *query_data, size_t k, const GenericFilter& conditions) const
        {
            std::priority_queue<std::pair<dist_t, labeltype>> result;
            std::vector<size_t> candidates;
            candidates.reserve(this->max_elements_ / 10);

            for (size_t i = 0; i < this->cur_element_count; i++)
            {
                if (conditions.check(i))
                {
                    candidates.push_back(i);
                }
            }

            for (size_t idx : candidates)
            {
                dist_t d = this->fstdistfunc_(query_data, this->getDataByInternalId(idx), this->dist_func_param_);
                if (result.size() < k)
                    result.push(std::pair<dist_t, labeltype>(d, this->getExternalLabel(idx)));
                else if (d < result.top().first)
                {
                    result.push(std::pair<dist_t, labeltype>(d, this->getExternalLabel(idx)));
                    result.pop();
                }
            }

            while (result.size() < k)
                result.push(std::pair<dist_t, labeltype>(1000000.0f, -1));
            return result;
        }

        float selectivityEstimator(const OptimizedFilter& conditions) const
        {
            if (this->max_elements_ == 0)
                return 0.0f;
                
            size_t count = 0;
            size_t step, sample_size;
            
            if (this->max_elements_ <= 1000) {
                step = 1;
                sample_size = this->max_elements_;
            } else if (this->max_elements_ <= 10000) {
                step = 10;
                sample_size = this->max_elements_ / 10;
            } else if (this->max_elements_ <= 100000) {
                step = 100;
                sample_size = this->max_elements_ / 100;
            } else {
                sample_size = 10000;
                step = this->max_elements_ / 10000;
                if (step < 1) step = 1;
            }
            
            for (size_t i = 0; i < this->max_elements_; i += step)
            {
                if (conditions.check(getAttributeByInternalId(i)))
                    count++;
            }
            
            if (sample_size == 0)
                return 0.0f;
            
            return static_cast<float>(count) / static_cast<float>(sample_size);
        }

        template <typename GenericFilter>
        float selectivityEstimatorGeneric(const GenericFilter& conditions) const
        {
            if (this->cur_element_count == 0)
                return 0.0f;

            size_t count = 0;
            size_t step, sample_size;

            if (this->cur_element_count <= 1000) {
                step = 1;
                sample_size = this->cur_element_count;
            } else if (this->cur_element_count <= 10000) {
                step = 10;
                sample_size = this->cur_element_count / 10;
            } else if (this->cur_element_count <= 100000) {
                step = 100;
                sample_size = this->cur_element_count / 100;
            } else {
                sample_size = 10000;
                step = this->cur_element_count / 10000;
                if (step < 1) step = 1;
            }

            for (size_t i = 0; i < this->cur_element_count; i += step)
            {
                if (conditions.check(i))
                    count++;
            }

            if (sample_size == 0)
                return 0.0f;

            return static_cast<float>(count) / static_cast<float>(sample_size);
        }

        std::priority_queue<std::pair<dist_t, labeltype>>
        searchKnn(const void *query_data, size_t k, std::vector<FilterConditionWithId> filtering_conditions) const
        {
            OptimizedFilter conditions(filtering_conditions);
            
            float p = selectivityEstimator(conditions);
            if (p > 0.01f)
                return searchGraph(query_data, k, p, conditions);
            else
                return searchBruteForce(query_data, k, conditions);
        }

        template <typename GenericFilter>
        std::priority_queue<std::pair<dist_t, labeltype>>
        searchKnnGeneric(const void *query_data, size_t k, const GenericFilter& conditions) const
        {
            float p = selectivityEstimatorGeneric(conditions);
            if (p > 0.01f)
                return searchGraphGeneric(query_data, k, p, conditions);
            return searchBruteForceGeneric(query_data, k, conditions);
        }

        template <typename GenericFilter>
        std::priority_queue<std::pair<dist_t, labeltype>>
        searchKnnGenericGraphOnly(const void *query_data, size_t k, float p, const GenericFilter& conditions) const
        {
            return searchGraphGeneric(query_data, k, p, conditions);
        }

        ~FAVOR()
        {
            this->clear();
        }
    };
}
