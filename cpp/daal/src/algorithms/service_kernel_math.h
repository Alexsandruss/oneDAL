/* file: service_kernel_math.h */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
//++
//  Implementation of math functions.
//--
*/

#ifndef __SERVICE_KERNEL_MATH_H__
#define __SERVICE_KERNEL_MATH_H__

#include "services/daal_defines.h"
#include "services/error_handling.h"
#include "src/data_management/service_numeric_table.h"
#include "src/services/service_data_utils.h"
#include "src/services/service_arrays.h"
#include "src/externals/service_blas.h"
#include "src/externals/service_lapack.h"
#include "src/externals/service_memory.h"
#include "src/externals/service_math.h"
#include "src/algorithms/svd/svd_dense_default_impl.i"
#include "mkl_lapacke.h"
#include <stdlib.h>

using namespace daal::internal;
using namespace daal::services;
using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu>
bool solveSystemWithPLU(size_t nCols, algorithmFPType * a, algorithmFPType * b)
{
    /* Fill upper side of symmetric matrix */
    for (size_t i = 1; i < nCols; ++i)
    {
        for (size_t j = 0; j < i; ++j)
        {
            a[j * nCols + i] = a[i * nCols + j];
        }
    }

    TArray<long, cpu> ipiv(nCols);
    int info = 0;

    /* Perform PLU decomposition: A = P*L*U */
    Lapack<algorithmFPType, cpu>::xgetrf(LAPACK_COL_MAJOR, nCols, nCols, a, nCols, ipiv.get(), &info);
    if (info != 0)
    {
        return false;
    }
    /* Solve P*L*U * x = b */
    Lapack<algorithmFPType, cpu>::xgetrs(LAPACK_COL_MAJOR, 'N', nCols, 1, a, nCols, ipiv.get(), b, nCols, &info);

    return (info == 0);
}

template <typename algorithmFPType, CpuType cpu>
bool solveSystemWithQR(size_t nCols, algorithmFPType * a, algorithmFPType * b)
{
    for (size_t i = 1; i < nCols; ++i)
    {
        for (size_t j = 0; j < i; ++j)
        {
            a[j * nCols + i] = a[i * nCols + j];
        }
    }

    TArray<algorithmFPType, cpu> rArr(nCols * nCols);
    for (size_t i = 0; i < nCols * nCols; ++i) rArr[i] = algorithmFPType(0.0);
    daal::algorithms::svd::internal::compute_QR_on_one_node_seq<algorithmFPType, cpu>(nCols, nCols, a, nCols, rArr.get(), nCols);

    DAAL_INT info;
    DAAL_INT iOne   = 1;
    DAAL_INT iNCols = nCols;
    algorithmFPType alpha = 1.0;
    algorithmFPType beta  = 0.0;
    DAAL_INT inc = 1;
    char tr = 'T';
    TArray<algorithmFPType, cpu> bArr(nCols);
    for (size_t i = 0; i < nCols; ++i) bArr[i] = algorithmFPType(0.0);
    Blas<algorithmFPType, cpu>::xxgemv(&tr, &iNCols, &iNCols, &alpha, a, &iNCols, b, &inc, &beta, bArr.get(), &inc);

    char up     = 'U';
    char nodiag = 'N';
    char trans  = 'N';
    Lapack<algorithmFPType, cpu>::xtrtrs(&up, &trans, &nodiag, &iNCols, &iOne, rArr.get(), &iNCols, bArr.get(), &iNCols, &info);
    daal::services::internal::daal_memcpy_s(b, nCols * sizeof(algorithmFPType), bArr.get(), nCols * sizeof(algorithmFPType));
    return (info == 0);
}

template <typename algorithmFPType, CpuType cpu>
bool solveSystem(size_t nCols, algorithmFPType * a, algorithmFPType * b)
{
    char* strategy;
    strategy = getenv("SOLUTION_STRATEGY");
    char def_strat = 'd';
    if (strategy == NULL)
    {
        strategy = &def_strat;
    }

    if (strategy[0] == 'Q')
    {
        return solveSystemWithQR<algorithmFPType, cpu>(nCols, a, b);
    }
    else if (strategy[0] == 'P')
    {
        return solveSystemWithPLU<algorithmFPType, cpu>(nCols, a, b);
    }

    DAAL_INT info = 0;
    /* Backup A for fallback to PLU decomposition */
    TArray<algorithmFPType, cpu> aCopy(nCols * nCols);
    info |= daal::services::internal::daal_memcpy_s(aCopy.get(), nCols * nCols * sizeof(algorithmFPType), a, nCols * nCols * sizeof(algorithmFPType));

    /* POTRF and POTRS parameters */
    char uplo     = 'U';
    DAAL_INT iOne = 1;

    /* Perform Cholesky decomposition: A = L*L' */
    Lapack<algorithmFPType, cpu>::xpotrf(&uplo, (DAAL_INT *)&nCols, a, (DAAL_INT *)&nCols, &info);
    if (info != 0)
    {
        if (strategy[0] == 'q')
        {
            return solveSystemWithQR<algorithmFPType, cpu>(nCols, aCopy.get(), b);
        }
        else if (strategy[0] == 'p')
        {
            return solveSystemWithPLU<algorithmFPType, cpu>(nCols, aCopy.get(), b);
        }
        return false;
    }

    /* Backup b for fallback to PLU decomposition */
    TArray<algorithmFPType, cpu> bCopy(nCols);
    info |= daal::services::internal::daal_memcpy_s(bCopy.get(), nCols * sizeof(algorithmFPType), b, nCols * sizeof(algorithmFPType));

    /* Solve L*L' * x = b */
    Lapack<algorithmFPType, cpu>::xpotrs(&uplo, (DAAL_INT *)&nCols, &iOne, a, (DAAL_INT *)&nCols, b, (DAAL_INT *)&nCols, &info);
    if (info != 0)
    {
        if (strategy[0] == 'q')
        {
            /* Fallback to QR decomposition if Cholesky is failed */
            if (solveSystemWithQR<algorithmFPType, cpu>(nCols, aCopy.get(), bCopy.get()))
            {
                info = daal::services::internal::daal_memcpy_s(b, nCols * sizeof(algorithmFPType), bCopy.get(), nCols * sizeof(algorithmFPType));
                return (info == 0);
            }
            else
            {
                return false;
            }
        }
        else if (strategy[0] == 'p')
        {
            /* Fallback to PLU decomposition if Cholesky is failed */
            if (solveSystemWithPLU<algorithmFPType, cpu>(nCols, aCopy.get(), bCopy.get()))
            {
                info = daal::services::internal::daal_memcpy_s(b, nCols * sizeof(algorithmFPType), bCopy.get(), nCols * sizeof(algorithmFPType));
                return (info == 0);
            }
            else
            {
                return false;
            }
        }
        return false;
    }
    return true;
}

template <typename FPType, CpuType cpu>
FPType distancePow2(const FPType * a, const FPType * b, size_t dim)
{
    FPType sum = 0.0;

    for (size_t i = 0; i < dim; i++)
    {
        sum += (b[i] - a[i]) * (b[i] - a[i]);
    }

    return sum;
}

template <typename FPType, CpuType cpu>
FPType distancePow(const FPType * a, const FPType * b, size_t dim, FPType p)
{
    FPType sum = 0.0;

    for (size_t i = 0; i < dim; i++)
    {
        sum += daal::internal::Math<FPType, cpu>::sPowx(b[i] - a[i], p);
    }

    return sum;
}

template <typename FPType, CpuType cpu>
FPType distance(const FPType * a, const FPType * b, size_t dim, FPType p)
{
    FPType sum = 0.0;

    for (size_t i = 0; i < dim; i++)
    {
        sum += daal::internal::Math<FPType, cpu>::sPowx(b[i] - a[i], p);
    }

    return daal::internal::Math<FPType, cpu>::sPowx(sum, (FPType)1.0 / p);
}

template <typename FPType, CpuType cpu>
class PairwiseDistances
{
public:
    virtual ~PairwiseDistances() {};

    virtual services::Status init() = 0;

    virtual services::Status computeBatch(const FPType * const a, const FPType * const b, size_t aOffset, size_t aSize, size_t bOffset, size_t bSize,
                                          FPType * const res)                                                             = 0;
    virtual services::Status computeBatch(size_t aOffset, size_t aSize, size_t bOffset, size_t bSize, FPType * const res) = 0;
    virtual services::Status computeFull(FPType * const res)                                                              = 0;
};

// compute: sum(A^2, 2) + sum(B^2, 2) -2*A*B'
template <typename FPType, CpuType cpu>
class EuclideanDistances : public PairwiseDistances<FPType, cpu>
{
public:
    EuclideanDistances(const NumericTable & a, const NumericTable & b, bool squared = true) : _a(a), _b(b), _squared(squared) {}

    virtual ~EuclideanDistances() {}

    virtual services::Status init()
    {
        services::Status s;

        normBufferA.reset(_a.getNumberOfRows());
        DAAL_CHECK_MALLOC(normBufferA.get());
        s |= computeNorm(_a, normBufferA.get());

        if (&_a != &_b)
        {
            normBufferB.reset(_b.getNumberOfRows());
            DAAL_CHECK_MALLOC(normBufferB.get());
            s |= computeNorm(_b, normBufferB.get());
        }

        return s;
    }

    // output:  Row-major matrix of size { aSize x bSize }
    virtual services::Status computeBatch(const FPType * const a, const FPType * const b, size_t aOffset, size_t aSize, size_t bOffset, size_t bSize,
                                          FPType * const res)
    {
        const size_t nRowsA = aSize;
        const size_t nColsA = _a.getNumberOfColumns();
        const size_t nRowsB = bSize;

        const size_t nRowsC = nRowsA;
        const size_t nColsC = nRowsB;

        computeABt(a, b, nRowsA, nColsA, nRowsB, res);

        const FPType * const aa = normBufferA.get() + aOffset;
        const FPType * const bb = (&_a == &_b) ? normBufferA.get() + bOffset : normBufferB.get() + bOffset;

        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t i = 0; i < nRowsC; i++)
        {
            for (size_t j = 0; j < nColsC; j++)
            {
                res[i * nColsC + j] = aa[i] + bb[j] - 2 * res[i * nColsC + j];
            }
        }

        if (!_squared)
        {
            daal::internal::Math<FPType, cpu> math;
            daal::services::internal::TArray<FPType, cpu> tmpArr(nRowsC * nColsC);
            FPType * tmp = tmpArr.get();
            math.vSqrt(nRowsC * nColsC, res, tmp);

            services::internal::daal_memcpy_s(res, nRowsC * nColsC * sizeof(FPType), tmp, nRowsC * nColsC * sizeof(FPType));
        }

        return services::Status();
    }

    // output:  Row-major matrix of size { aSize x bSize }
    virtual services::Status computeBatch(size_t aOffset, size_t aSize, size_t bOffset, size_t bSize, FPType * const res)
    {
        ReadRows<FPType, cpu> aDataRows(const_cast<NumericTable *>(&_a), aOffset, aSize);
        DAAL_CHECK_BLOCK_STATUS(aDataRows);
        const FPType * const aData = aDataRows.get();

        ReadRows<FPType, cpu> bDataRows(const_cast<NumericTable *>(&_b), bOffset, bSize);
        DAAL_CHECK_BLOCK_STATUS(bDataRows);
        const FPType * const bData = bDataRows.get();

        return computeBatch(aData, bData, aOffset, aSize, bOffset, bSize, res);
    }

    // output:  Row-major matrix of size { nrows(A) x nrows(B) }
    virtual services::Status computeFull(FPType * const res)
    {
        SafeStatus safeStat;

        const size_t nRowsA    = _a.getNumberOfRows();
        const size_t blockSize = 256;
        const size_t nBlocks   = nRowsA / blockSize + (nRowsA % blockSize > 0);

        daal::threader_for(nBlocks, nBlocks, [&](size_t iBlock) {
            const size_t i1    = iBlock * blockSize;
            const size_t i2    = (iBlock + 1 == nBlocks ? nRowsA : i1 + blockSize);
            const size_t iSize = i2 - i1;

            const size_t nRowsB = _b.getNumberOfRows();

            DAAL_CHECK_STATUS_THR(computeBatch(i1, iSize, 0, nRowsB, res + i1 * nRowsB));
        });

        return safeStat.detach();
    }

protected:
    // compute (sum(A^2, 2))
    services::Status computeNorm(const NumericTable & ntData, FPType * const res)
    {
        const size_t nRows = ntData.getNumberOfRows();
        const size_t nCols = ntData.getNumberOfColumns();

        const size_t blockSize = 512;
        const size_t nBlocks   = nRows / blockSize + !!(nRows % blockSize);

        SafeStatus safeStat;

        daal::threader_for(nBlocks, nBlocks, [&](size_t iBlock) {
            size_t begin = iBlock * blockSize;
            size_t end   = services::internal::min<cpu, size_t>(begin + blockSize, nRows);

            ReadRows<FPType, cpu> dataRows(const_cast<NumericTable &>(ntData), begin, end - begin);
            DAAL_CHECK_BLOCK_STATUS_THR(dataRows);
            const FPType * const data = dataRows.get();

            FPType * r = res + begin;

            for (size_t i = 0; i < end - begin; i++)
            {
                FPType sum = FPType(0);
                PRAGMA_IVDEP
                PRAGMA_ICC_NO16(omp simd reduction(+ : sum))
                for (size_t j = 0; j < nCols; j++)
                {
                    sum += data[i * nCols + j] * data[i * nCols + j];
                }
                r[i] = sum;
            }
        });

        return safeStat.detach();
    }

    // compute (A x B')
    void computeABt(const FPType * const a, const FPType * const b, const size_t nRowsA, const size_t nColsA, const size_t nRowsB, FPType * const out)
    {
        const char transa    = 't';
        const char transb    = 'n';
        const DAAL_INT _m    = nRowsB;
        const DAAL_INT _n    = nRowsA;
        const DAAL_INT _k    = nColsA;
        const FPType alpha   = 1.0;
        const DAAL_INT lda   = nColsA;
        const DAAL_INT ldy   = nColsA;
        const FPType beta    = 0.0;
        const DAAL_INT ldaty = nRowsB;

        Blas<FPType, cpu>::xxgemm(&transa, &transb, &_m, &_n, &_k, &alpha, b, &lda, a, &ldy, &beta, out, &ldaty);
    }

    const NumericTable & _a;
    const NumericTable & _b;
    const bool _squared;

    TArray<FPType, cpu> normBufferA;
    TArray<FPType, cpu> normBufferB;
};

} // namespace internal
} // namespace algorithms
} // namespace daal

#endif
