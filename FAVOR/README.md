# FAVOR

This repository contains the official implementation of FAVOR, a novel vector search system for hybrid queries that combine approximate nearest neighbor search (ANNS) with complex attribute filtering.

FAVOR efficiently handles arbitrary filtering conditions and maintains stable, high performance across different selectivity levels. It features an integrated architecture, a new HNSW-based search algorithm with an exclusion mechanism, and a dynamic search selector.

## Installation

### Requirements

- C++17 compiler (GCC 7+ / Clang 5+)
- CMake 3.12+
- OpenMP 4.0+ (optional, for parallelism)

### Build from Source

```bash
mkdir build && cd build
cmake ..
make -j && cd ..
```

## Usage

FAVOR provides a complete workflow for Filtered ANNS: generate attributes → build index → search with filters. We include a small `Words` dataset in the `data/` directory for quick testing. The main steps are:

### 1. Generate Attribute File

Generate attribute data for your vector dataset:

```bash
./build/app/generate_attribute <baseset_path> <attribute_path>
```

**Example:**
```bash
./build/app/generate_attribute data/words_base.fvecs data/attribute.txt
```

### 2. Generate Query Conditions

Generate filtering conditions for queries:

```bash
./build/app/generate_query_conditions <queryset_path> <condition_path>
```

**Example:**
```bash
./build/app/generate_query_conditions data/words_query.fvecs data/conditions.txt
```

### 3. Generate Ground Truth

Generate ground truth for evaluation:

```bash
./build/app/generate_groundtruth <baseset_path> <queryset_path> <attribute_path> <topk> <groundtruth_path> <condition_path>
```

**Example:**
```bash
./build/app/generate_groundtruth \
    data/words_base.fvecs \
    data/words_query.fvecs \
    data/attribute.txt \
    10 \
    data/groundtruth.bin \
    data/conditions.txt
```

### 4. Build Index

Build the FAVOR index:

```bash
./build/app/build_index <baseset_path> <attribute_path> <index_path>
```

**Example:**
```bash
./build/app/build_index data/words_base.fvecs data/attribute.txt data/index.bin
```

### 5. Search

Perform filtered vector search:

```bash
./build/app/search <baseset_path> <queryset_path> <attribute_path> <topk> <groundtruth_path> <condition_path> <index_path> <ef>
```

**Example:**
```bash
./build/app/search \
    data/words_base.fvecs \
    data/words_query.fvecs \
    data/attribute.txt \
    10 \
    data/groundtruth.bin \
    data/conditions.txt \
    data/index.bin \
    100
```

### Filter Expression Syntax

FAVOR supports complex filter expressions with the following operators:

| Operator | Description | Example |
|----------|-------------|---------|
| `==` | Equal to | `color == 1` |
| `!=` | Not equal to | `color != 2` |
| `<` | Less than | `age < 30` |
| `<=` | Less than or equal | `price <= 100` |
| `>` | Greater than | `age > 25` |
| `>=` | Greater than or equal | `rating >= 4.5` |
| `AND` | Logical AND | `color == 1 AND price < 50` |
| `IN` | Set membership | `label IN [1, 3, 4]` |

### Condition File Format

The `condition_path` file contains one filter expression per line, with each line corresponding to a query vector. Example:

```
color == 1
color != 2
color IN [1, 3, 4]
color > 50
color < 25
```

### Data Formats

#### Vector Data (.fvecs)

Binary format where each vector is stored as:
- 4 bytes: dimension (int32, little-endian)
- dimension × 4 bytes: vector components (float32, little-endian)

#### Attribute File (.txt)

Text format:
```
{num_vectors}
{num_attributes}
{attribute_name_1}
{value_1}
{value_2}
...
{attribute_name_2}
{value_1}
{value_2}
...
```

#### Ground Truth (.bin)

Binary format containing top-k nearest neighbor IDs for each query vector.

#### Index (.bin)

Binary format containing the FAVOR index structure.