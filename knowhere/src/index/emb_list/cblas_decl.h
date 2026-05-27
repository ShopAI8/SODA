// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

// Prefer the system/vendor CBLAS headers when available to avoid duplicate
// enum/function declarations (e.g. MKL/OpenBLAS conflicts). Fallback to local
// declarations only when neither cblas.h nor mkl_cblas.h can be found.

#ifndef KNOWHERE_CBLAS_DECL_H
#define KNOWHERE_CBLAS_DECL_H

#if defined(CBLAS_H) || defined(MKL_CBLAS_H)
// CBLAS is already included by another header.
#elif defined(__has_include)
#if __has_include(<cblas.h>)
#include <cblas.h>
#elif __has_include(<mkl_cblas.h>)
#include <mkl_cblas.h>
#else
extern "C" {
enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 };
enum CBLAS_SIDE { CblasLeft = 141, CblasRight = 142 };
enum CBLAS_UPLO { CblasUpper = 121, CblasLower = 122 };
enum CBLAS_DIAG { CblasNonUnit = 131, CblasUnit = 132 };
void cblas_sgemm(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB, int M, int N,
                 int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C,
                 int ldc);
void cblas_strsm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side, enum CBLAS_UPLO Uplo, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_DIAG Diag, int M, int N, float alpha, const float* A, int lda, float* B, int ldb);
void cblas_saxpy(int N, float alpha, const float* X, int incX, float* Y, int incY);
}  // extern "C"
#endif
#else
extern "C" {
enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 };
enum CBLAS_SIDE { CblasLeft = 141, CblasRight = 142 };
enum CBLAS_UPLO { CblasUpper = 121, CblasLower = 122 };
enum CBLAS_DIAG { CblasNonUnit = 131, CblasUnit = 132 };
void cblas_sgemm(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA, enum CBLAS_TRANSPOSE TransB, int M, int N,
                 int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C,
                 int ldc);
void cblas_strsm(enum CBLAS_ORDER Order, enum CBLAS_SIDE Side, enum CBLAS_UPLO Uplo, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_DIAG Diag, int M, int N, float alpha, const float* A, int lda, float* B, int ldb);
void cblas_saxpy(int N, float alpha, const float* X, int incX, float* Y, int incY);
}  // extern "C"
#endif

#endif  // KNOWHERE_CBLAS_DECL_H
