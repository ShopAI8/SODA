#ifndef LABEL_NAV_GRAPH_H
#define LABEL_NAV_GRAPH_H

#include <vector>
#include <unordered_set>
#include "config.h"

namespace ANNS
{

   class LabelNavGraph
   {

   public:
      LabelNavGraph(IdxType num_nodes)
      {
         in_neighbors.resize(num_nodes + 1);
         out_neighbors.resize(num_nodes + 1);
         coverage_ratio.resize(num_nodes + 1, 0.0); // Coverage ratio for each node
         covered_sets.resize(num_nodes + 1);        // Covered vector set for each node
         in_degree.resize(num_nodes + 1, 0);
         out_degree.resize(num_nodes + 1, 0);

         _lng_descendants_num.resize(num_nodes + 1); // group_id, descendants_count
      };

      std::vector<std::vector<IdxType>> in_neighbors, out_neighbors;
      std::vector<double> coverage_ratio;                    // Coverage ratio for each label set
      std::vector<std::unordered_set<IdxType>> covered_sets; // Covered vector set for each group
      std::vector<int> in_degree, out_degree;                // In-degree and out-degree

      std::vector<std::pair<IdxType, int>> _lng_descendants_num; // group_id, descendants_count
      double avg_descendants;                                    // Average number of descendants
      std::vector<std::unordered_set<IdxType>> _lng_descendants; // Covered group set for each group
      ~LabelNavGraph() = default;

   private:
   };
}

#endif // LABEL_NAV_GRAPH_H
