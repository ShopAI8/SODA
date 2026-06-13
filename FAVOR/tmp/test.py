import csv
from collections import defaultdict

path = "/noraiddata/lijiakang/FilterVector/FilterVectorResults/Genome/Results/FAVOR/
Index[M32_LB100_alpha1.2_C6_EP16_AN108077_AM32_AMB64_AG80]_GT[GT_query_select_200_A_B_C-weighted-sub-base-
123456789_random_300_K10]_Search[Ls200-Le1000-Lp100_efsS100-efss100-efsf100-lt5000_K10_th10]/results/
query_details_repeat1.csv"

groups = defaultdict(lambda: {"n": 0, "time": 0.0, "recall": 0.0})

with open(path, newline="") as f:
     reader = csv.DictReader(f)
     for row in reader:
         key = (int(float(row["Lsearch"])), int(float(row["efs"])))
         groups[key]["n"] += 1
         groups[key]["time"] += float(row["Time_ms"])
         groups[key]["recall"] += float(row["Recall"])

print("Lsearch,efs,Recall,QPS,Time_ms")
for (lsearch, efs), g in sorted(groups.items()):
    time_ms = g["time"] / g["n"]
    recall = g["recall"] / g["n"]
    qps = 1000.0 / time_ms
    print(f"{lsearch},{efs},{recall:.6f},{qps:.6f},{time_ms:.6f}")
