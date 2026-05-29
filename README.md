# Filtered Approximate Nearest Neighbor Search on Vectors with Diverse Labels

This repository provides **SmartRoute**, an enhanced implementation for **Filtered Approximate Nearest Neighbor Search (Filtered ANNS)**. SmartRoute improves **UNG (Unified Navigating Graph)** with an accelerated variant named **UNG+**, and further integrates the core ideas of **UNG+**, **NaviX**, and **Pre-Filtering**. It introduces a new **AI-based intelligent routing** mechanism that adaptively schedules algorithms using a machine learning model. In addition, the improved version, **SmartRoute+**, addresses cache thrashing caused by frequent switching among algorithms.

---

## 1. Preparation

### 1.1 Environment Setup

We conduct experiments on a Linux server with two Intel Xeon Gold 6342 processors, 144 threads, and 1 TB RAM. All algorithms are implemented in C++ and compiled with GCC 11.4.0. The project is built with CMake 4.3.2 and Boost 1.85.0.

The experimental environment is as follows:

- CPU: 2 × Intel Xeon Gold 6342 processors
- Threads: 144
- Memory: 1 TB RAM
- Compiler: GCC 11.4.0
- CMake: 4.3.2
- Boost: 1.85.0
- Python: 3.10.20

Configure the Python environment for the experiments using Conda.

Run the following command to create the required environment:

```bash
conda env create -f environment.yml
```

### 1.2 Data Preparation

Download the required datasets from [Hugging Face](https://huggingface.co/datasets/Paper4Review/SmartRoute_data) into the `data` folder in advance.

Please note that `data` is the default directory for storing datasets. The dataset collection includes eight datasets: Amazon, BookReviews, Genome, Laion, Music, Reviews, Tiktok, and VariousImg. Please store each dataset under the `data` directory using the same directory structure.

Each dataset contains the following files:

1. `*_random_300/`: This folder contains the query files used for testing. The wildcard `*` represents the dataset name.
2. `*_base_labels.txt`: This file contains the ground-truth labels corresponding to the base vectors.
3. `*_base.bin`: This binary file contains the base vectors of the dataset.
4. `*_base.fvecs`: This file stores the base vectors in `fvecs` format.

### 1.3 Repository Structure

The main structure of the repository is as follows:

```text
FilterVectorCode/
├── ACORN/                  # Implementation related to ACORN
├── NaviX/                  # Implementation related to NaviX
├── UNG/                    # UNG / UNG+ implementation and data-processing scripts
├── knowhere/               # Dependency for the Milvus baseline
├── data/                   # Dataset directory
├── experiment_json/        # Example experiment configurations
├── build_hybrid.sh         # Unified build script
├── generate_gt.sh          # Ground-truth generation script
├── search.sh               # Search and result aggregation script
├── exp.sh                  # Main experiment entry script
├── final_exp.sh            # Supplementary experiment entry script
├── generate_queries.sh     # Query generation script
├── generate_queries_config.json
└── environment.yml         # Recommended Conda environment
```

## 2. Experiments

Experiment configurations are stored in the `experiment_json/` directory.

### 2.1 Running Experiments

You can modify the experimental parameters and dataset paths in the corresponding configuration files before running the experiments.

Run the following command to start an experiment:

```bash
./exp.sh experiment_json/experiments-Genome-200-random-300-mix-len.json
```

### 2.2 Experiment Workflow

1. **Configuration Parsing**: `exp.sh` reads the `experiments` array from the JSON configuration file.
2. **Index Construction**: `build_hybrid.sh` is invoked to build indexes. The script supports the `parallel` mode, which builds the UNG index and ACORN index simultaneously.
3. **Ground-Truth Generation**: `generate_gt.sh` is invoked to compute the true nearest neighbors, which are used for recall evaluation.
4. **Search Execution**: `search.sh` is invoked to run the search process. It loads the pretrained ONNX router model for online scheduling.

---

## 3. Core Parameters

The following parameters in the configuration files or scripts determine the behavior of the algorithms:

| Parameter | Description | Values / Notes |
| :--- | :--- | :--- |
| `ROUTING_MODE` | Determines the routing logic. | `0`: Baseline mode; `1`: **SmartRoute**; `5`: **SmartRoute+**. |
| `BASELINE_ALG` | Specifies the algorithm when `ROUTING_MODE=0`. | `0`: UNG; `2`: ACORN-gamma; `4`: NaviX; `5`: Pre-Filtering; `6`: ACORN-1; `8`: UNG+; `9`: Milvus-IVF; `10`: Milvus-HNSW. |
| `BUILD_MODE` | Specifies the index construction mode. | `parallel`: Build all indexes in parallel; `acorn_only`: Build only the ACORN index. |
| `Lsearch` | Search parameter for UNG. | Similar to `efSearch` in HNSW; controls the search depth. |
| `efs_start/step` | Search parameters for ACORN/NaviX. | Used to dynamically adjust the filtering strength during search. |

---

## 4. Model Training and Deployment

The decision model for intelligent routing is trained using Python scripts:

- **Training**: Use `selector/smart_route_train.py`.
- **Deployment**: Export the trained model to `.onnx` format and place it in the `SelectModels` directory, where it can be loaded by the C++ `MethodSelector`.