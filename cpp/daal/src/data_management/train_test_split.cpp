/** file train_test_split.cpp */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "data_management/data/internal/train_test_split.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/data_dictionary.h"
#include "services/env_detect.h"
#include "src/externals/service_dispatch.h"
#include "src/services/service_data_utils.h"
#include "src/services/service_utils.h"
#include "src/services/service_defines.h"
#include "src/threading/threading.h"
#include "src/data_management/service_numeric_table.h"
#include "data_management/features/defines.h"
#include "src/algorithms/service_error_handling.h"
#include "src/externals/service_rng.h"
#include "src/externals/service_rng_mkl.h"
#include "algorithms/engines/mt19937/mt19937.h"
#include "algorithms/distributions/bernoulli/bernoulli.h"
#include <chrono>

using namespace std::chrono;

using namespace daal::algorithms;
using namespace daal::data_management;
using namespace daal::algorithms::engines;
using namespace daal::algorithms::distributions;

namespace daal
{
namespace data_management
{
namespace internal
{
typedef daal::data_management::NumericTable::StorageLayout NTLayout;

const size_t BLOCK_CONST      = 2048;
const size_t THREADING_BORDER = 8388608;

template <typename IdxType, daal::CpuType cpu>
services::Status generateShuffledIndicesImpl(const NumericTablePtr & idxTable, const unsigned int seed)
{
    high_resolution_clock::time_point t0 = high_resolution_clock::now();

    const size_t nThreads = threader_get_threads_number();
    const size_t n        = idxTable->getNumberOfRows();
    daal::internal::WriteColumns<IdxType, cpu> idxBlock(*idxTable, 0, 0, n);
    IdxType * idx = idxBlock.get();
    DAAL_CHECK_MALLOC(idx);

    daal::services::internal::TArray<float, cpu> bernDistrArr(n);
    float * bernDistr = bernDistrArr.get();
    DAAL_CHECK_MALLOC(bernDistr);

    const size_t blockSize = 16 * BLOCK_CONST;
    const size_t nBlocks   = n / blockSize + !!(n % blockSize);

    const double p = 0.1;
    size_t nTrainNeeded, nTestNeeded, nTrain, nTest;
    nTestNeeded = n * p;
    nTrainNeeded = n - nTestNeeded;
    // printf("train size - %f, n - %lu, nTrainNeeded - %lu, nTestNeeded - %lu\n", p, n, nTrainNeeded, nTestNeeded);

    daal::services::internal::TArray<int, cpu> nSubTestArr(nBlocks);
    daal::services::internal::TArray<int, cpu> nSubTrainArr(nBlocks);

    int * nSubTest = nSubTestArr.get();
    int * nSubTrain = nSubTrainArr.get();
    DAAL_CHECK_MALLOC(nSubTest);
    DAAL_CHECK_MALLOC(nSubTrain);

    for (size_t iBlock = 0; iBlock < nBlocks; ++iBlock)
    {
        nSubTest[iBlock] = 0;
        nSubTrain[iBlock] = 0;
    }

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    daal::threader_for(nBlocks, nBlocks, [&](size_t iBlock) {
        const size_t start = iBlock * blockSize;
        const size_t end   = daal::services::internal::min<cpu, size_t>(start + blockSize, n);
        // daal::internal::BaseRNGs<cpu> baseRng(seed, VSL_BRNG_MT19937);
        // baseRng.skipAhead(start);
        // daal::internal::RNGs<int, cpu> rng;

        // rng.bernoulli(end - start, bernDistr + start, baseRng.getState(), p);

        NumericTablePtr dataTable(new HomogenNumericTable<float>(bernDistr + start, end - start, 1));
        bernoulli::Batch<float> bernoulli(p);
        bernoulli.input.set(distributions::tableToFill, dataTable);
        services::SharedPtr<mt19937::Batch<float>> eng = mt19937::Batch<float>::create(seed);
        eng->skipAhead(start);
        bernoulli.parameter.engine = eng;
        bernoulli.compute();

        for (size_t i = start; i < end; ++i)
        {
            if (bernDistr[i] == 0)
            {
                ++nSubTrain[iBlock];
            }
            else
            {
                ++nSubTest[iBlock];
            }
        }
    });

    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    nTrain = 0;
    nTest = 0;
    for (size_t iBlock = 0; iBlock < nBlocks; ++iBlock)
    {
        nTrain += nSubTrain[iBlock];
        nTest += nSubTest[iBlock];
    }

    // printf("unbalanced ns: %lu %lu\n", nTrain, nTest);

    high_resolution_clock::time_point t3 = high_resolution_clock::now();

    size_t skipTrain = 0;
    size_t skipTest = 0;
    while ( nTrain < nTrainNeeded )
    {
        // printf("balancing train: %lu of %lu\n", nTrain, nTrainNeeded);
        size_t nGen = (nTrainNeeded - nTrain) * 4;
        if (nGen <= 0)
            break;

        daal::services::internal::TArray<IdxType, cpu> addIdxArr(nGen);
        IdxType * addIdx = addIdxArr.get();
        DAAL_CHECK_MALLOC(addIdx);

        daal::internal::BaseRNGs<cpu> baseRng(seed, VSL_BRNG_MT19937);
        baseRng.skipAhead(skipTrain);
        daal::internal::RNGs<IdxType, cpu> rng;

        rng.uniform(nGen, addIdx, baseRng.getState(), 0, n);

        for (size_t i = 0; i < nGen; ++i)
        {
            if ( bernDistr[addIdx[i]] == 1 )
            {
                bernDistr[addIdx[i]] = 0;
                ++nTrain;
                --nTest;
            }
            if (nTrain == nTrainNeeded)
                break;
        }
        skipTrain += nGen;
    }

    high_resolution_clock::time_point t4 = high_resolution_clock::now();

    while ( nTest < nTestNeeded )
    {
        // printf("balancing test: %lu of %lu\n", nTest, nTestNeeded);
        size_t nGen = (nTestNeeded - nTest) * 4;
        if (nGen <= 0)
            break;

        daal::services::internal::TArray<IdxType, cpu> addIdxArr(nGen);
        IdxType * addIdx = addIdxArr.get();
        DAAL_CHECK_MALLOC(addIdx);

        daal::internal::BaseRNGs<cpu> baseRng(seed, VSL_BRNG_MT19937);
        baseRng.skipAhead(skipTrain + skipTest);
        daal::internal::RNGs<IdxType, cpu> rng;

        rng.uniform(nGen, addIdx, baseRng.getState(), 0, n);

        for (size_t i = 0; i < nGen; ++i)
        {
            if ( bernDistr[addIdx[i]] == 0 )
            {
                bernDistr[addIdx[i]] = 1;
                ++nTest;
                --nTrain;
            }
            if (nTest == nTestNeeded)
                break;
        }
        skipTest += nGen;
    }

    // printf("balanced ns: %lu %lu\n", nTrain, nTest);

    high_resolution_clock::time_point t5 = high_resolution_clock::now();

    size_t iTrain = 0;
    size_t iTest = nTrain;
    for (size_t i = 0; i < n; ++i)
    {
        if (bernDistr[i] == 0)
        {
            idx[iTrain] = i;
            ++iTrain;
        }
        else
        {
            idx[iTest] = i;
            ++iTest;
        }
    }
    // for (size_t i = 0; i < n; ++i)
    // {
    //     if (bernDistr[i] == 1)
    //     {
    //         idx[j] = i;
    //         ++j;
    //     }
    // }

    high_resolution_clock::time_point t6 = high_resolution_clock::now();

    size_t dt0 = duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    size_t dt1 = duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    size_t dt2 = duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
    size_t dt3 = duration_cast<std::chrono::nanoseconds>(t4 - t3).count();
    size_t dt4 = duration_cast<std::chrono::nanoseconds>(t5 - t4).count();
    size_t dt5 = duration_cast<std::chrono::nanoseconds>(t6 - t5).count();

    // printf("idx:init       - %lu\n", dt0);
    // printf("idx:bernDistr  - %lu\n", dt1);
    // printf("idx:nCounts    - %lu\n", dt2);
    // printf("idx:trainFix   - %lu\n", dt3);
    // printf("idx:testFix    - %lu\n", dt4);
    // printf("idx:assign     - %lu\n", dt5);

    return services::Status();
}

template <typename IdxType>
void generateShuffledIndicesDispImpl(const NumericTablePtr & idxTable, const unsigned int seed)
{
#define DAAL_GENERATE_INDICES(cpuId, ...) generateShuffledIndicesImpl<IdxType, cpuId>(__VA_ARGS__);
    DAAL_DISPATCH_FUNCTION_BY_CPU(DAAL_GENERATE_INDICES, idxTable, seed);
#undef DAAL_GENERATE_INDICES
}

template <typename IdxType>
DAAL_EXPORT void generateShuffledIndices(const NumericTablePtr & idxTable, const unsigned int seed)
{
    DAAL_SAFE_CPU_CALL((generateShuffledIndicesDispImpl<IdxType>(idxTable, seed)),
                       (generateShuffledIndicesImpl<IdxType, daal::CpuType::sse2>(idxTable, seed)));
}

template DAAL_EXPORT void generateShuffledIndices<int>(const NumericTablePtr & idxTable, const unsigned int seed);

template <typename DataType, typename IdxType, daal::CpuType cpu>
services::Status assignColumnValues(const DataType * origDataPtr, const NumericTablePtr & dataTable, const IdxType * idxPtr, const size_t startRow,
                                    const size_t nRows, const size_t iCol)
{
    daal::internal::WriteColumns<DataType, cpu> dataBlock(*dataTable, iCol, startRow, nRows);
    DataType * dataPtr = dataBlock.get();
    DAAL_CHECK_MALLOC(dataPtr);

    PRAGMA_IVDEP
    PRAGMA_VECTOR_ALWAYS
    for (size_t i = 0; i < nRows; ++i)
    {
        dataPtr[i] = origDataPtr[idxPtr[i]];
    }

    return services::Status();
}

template <typename DataType, typename IdxType, daal::CpuType cpu>
services::Status assignColumnSubset(const DataType * origDataPtr, const NumericTablePtr & dataTable, const IdxType * idxPtr, const size_t nRows,
                                    const size_t iCol, const size_t nThreads)
{
    if (nRows > THREADING_BORDER && nThreads > 1)
    {
        daal::SafeStatus s;
        const size_t nBlocks = nRows / BLOCK_CONST + !!(nRows % BLOCK_CONST);

        daal::threader_for(nBlocks, nBlocks, [&](size_t iBlock) {
            const size_t start = iBlock * BLOCK_CONST;
            const size_t end   = daal::services::internal::min<cpu, size_t>(start + BLOCK_CONST, nRows);

            s |= assignColumnValues<DataType, IdxType, cpu>(origDataPtr, dataTable, idxPtr, start, end - start, iCol);
        });
        return s.detach();
    }
    else
    {
        return assignColumnValues<DataType, IdxType, cpu>(origDataPtr, dataTable, idxPtr, 0, nRows, iCol);
    }
}

template <typename DataType, typename IdxType, daal::CpuType cpu>
services::Status splitColumn(const NumericTablePtr & inputTable, const NumericTablePtr & trainTable, const NumericTablePtr & testTable,
                             const IdxType * trainIdx, const IdxType * testIdx, const size_t nTrainRows, const size_t nTestRows, const size_t iCol,
                             const size_t nThreads)
{
    services::Status s;
    daal::internal::ReadColumns<DataType, cpu> origDataBlock(*inputTable, iCol, 0, nTrainRows + nTestRows);
    const DataType * origDataPtr = origDataBlock.get();
    DAAL_CHECK_MALLOC(origDataPtr);

    s |= assignColumnSubset<DataType, IdxType, cpu>(origDataPtr, trainTable, trainIdx, nTrainRows, iCol, nThreads);
    s |= assignColumnSubset<DataType, IdxType, cpu>(origDataPtr, testTable, testIdx, nTestRows, iCol, nThreads);

    return s;
}

template <typename DataType, typename IdxType, daal::CpuType cpu>
services::Status assignRows(const DataType * origDataPtr, const NumericTablePtr & dataTable, const NumericTablePtr & idxTable, const size_t startRow,
                            const size_t nRows, const size_t nColumns)
{
    daal::internal::WriteRows<DataType, cpu> dataBlock(*dataTable, startRow, nRows);
    daal::internal::ReadColumns<IdxType, cpu> idxBlock(*idxTable, 0, startRow, nRows);
    DataType * dataPtr     = dataBlock.get();
    const IdxType * idxPtr = idxBlock.get();
    DAAL_CHECK_MALLOC(dataPtr);
    DAAL_CHECK_MALLOC(idxPtr);

    for (size_t i = 0; i < nRows; ++i)
    {
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t j = 0; j < nColumns; ++j)
        {
            dataPtr[i * nColumns + j] = origDataPtr[idxPtr[i] * nColumns + j];
        }
    }

    return services::Status();
}

template <typename DataType, typename IdxType, daal::CpuType cpu>
services::Status assignRowsSubset(const DataType * origDataPtr, const NumericTablePtr & dataTable, const NumericTablePtr & idxTable,
                                  const size_t nRows, const size_t nColumns, const size_t nThreads, const size_t blockSize)
{
    if (nRows * nColumns > THREADING_BORDER && nThreads > 1)
    {
        daal::SafeStatus s;
        const size_t nBlocks = nRows / blockSize + !!(nRows % blockSize);

        daal::threader_for(nBlocks, nBlocks, [&](size_t iBlock) {
            const size_t start = iBlock * blockSize;
            const size_t end   = daal::services::internal::min<cpu, size_t>(start + blockSize, nRows);

            s |= assignRows<DataType, IdxType, cpu>(origDataPtr, dataTable, idxTable, start, end - start, nColumns);
        });
        return s.detach();
    }
    else
    {
        return assignRows<DataType, IdxType, cpu>(origDataPtr, dataTable, idxTable, 0, nRows, nColumns);
    }
}

template <typename DataType, typename IdxType, daal::CpuType cpu>
services::Status splitRows(const NumericTablePtr & inputTable, const NumericTablePtr & trainTable, const NumericTablePtr & testTable,
                           const NumericTablePtr & trainIdxTable, const NumericTablePtr & testIdxTable, const size_t nTrainRows,
                           const size_t nTestRows, const size_t nColumns, const size_t nThreads)
{
    services::Status s;
    const size_t blockSize = daal::services::internal::max<cpu, size_t>(BLOCK_CONST / nColumns, 1);
    daal::internal::ReadRows<DataType, cpu> origBlock(*inputTable, 0, nTrainRows + nTestRows);
    const DataType * origDataPtr = origBlock.get();
    DAAL_CHECK_MALLOC(origDataPtr);

    s |= assignRowsSubset<DataType, IdxType, cpu>(origDataPtr, trainTable, trainIdxTable, nTrainRows, nColumns, nThreads, blockSize);
    s |= assignRowsSubset<DataType, IdxType, cpu>(origDataPtr, testTable, testIdxTable, nTestRows, nColumns, nThreads, blockSize);

    return s;
}

template <typename IdxType, daal::CpuType cpu>
services::Status trainTestSplitImpl(const NumericTablePtr & inputTable, const NumericTablePtr & trainTable, const NumericTablePtr & testTable,
                                    const NumericTablePtr & trainIdxTable, const NumericTablePtr & testIdxTable)
{
    const size_t nThreads   = threader_get_threads_number();
    const NTLayout layout   = inputTable->getDataLayout();
    const size_t nColumns   = trainTable->getNumberOfColumns();
    const size_t nTrainRows = trainTable->getNumberOfRows();
    const size_t nTestRows  = testTable->getNumberOfRows();
    DAAL_OVERFLOW_CHECK_BY_MULTIPLICATION(size_t, nTrainRows + nTestRows, nColumns);

    NumericTableDictionaryPtr tableFeaturesDict = inputTable->getDictionarySharedPtr();

    if (layout == NTLayout::soa)
    {
        daal::SafeStatus s;
        daal::internal::ReadColumns<IdxType, cpu> trainIdxBlock(*trainIdxTable, 0, 0, nTrainRows);
        daal::internal::ReadColumns<IdxType, cpu> testIdxBlock(*testIdxTable, 0, 0, nTestRows);
        const IdxType * trainIdx = trainIdxBlock.get();
        const IdxType * testIdx  = testIdxBlock.get();
        DAAL_CHECK_MALLOC(trainIdx);
        DAAL_CHECK_MALLOC(testIdx);

        daal::conditional_threader_for(
            nColumns > 1 && nColumns * (nTrainRows + nTestRows) > THREADING_BORDER && nThreads > 1, nColumns, [&](size_t iCol) {
                switch ((*tableFeaturesDict)[iCol].getIndexType())
                {
                case daal::data_management::features::IndexNumType::DAAL_FLOAT32:
                    s |=
                        splitColumn<float, IdxType, cpu>(inputTable, trainTable, testTable, trainIdx, testIdx, nTrainRows, nTestRows, iCol, nThreads);
                    break;
                case daal::data_management::features::IndexNumType::DAAL_FLOAT64:
                    s |= splitColumn<double, IdxType, cpu>(inputTable, trainTable, testTable, trainIdx, testIdx, nTrainRows, nTestRows, iCol,
                                                           nThreads);
                    break;
                default:
                    s |= splitColumn<int, IdxType, cpu>(inputTable, trainTable, testTable, trainIdx, testIdx, nTrainRows, nTestRows, iCol, nThreads);
                }
            });
        return s.detach();
    }
    else
    {
        switch ((*tableFeaturesDict)[0].getIndexType())
        {
        case daal::data_management::features::IndexNumType::DAAL_FLOAT32:
            return splitRows<float, IdxType, cpu>(inputTable, trainTable, testTable, trainIdxTable, testIdxTable, nTrainRows, nTestRows, nColumns,
                                                  nThreads);
            break;
        case daal::data_management::features::IndexNumType::DAAL_FLOAT64:
            return splitRows<double, IdxType, cpu>(inputTable, trainTable, testTable, trainIdxTable, testIdxTable, nTrainRows, nTestRows, nColumns,
                                                   nThreads);
            break;
        default:
            return splitRows<int, IdxType, cpu>(inputTable, trainTable, testTable, trainIdxTable, testIdxTable, nTrainRows, nTestRows, nColumns,
                                                nThreads);
        }
    }
}

template <typename IdxType>
void trainTestSplitDispImpl(const NumericTablePtr & inputTable, const NumericTablePtr & trainTable, const NumericTablePtr & testTable,
                            const NumericTablePtr & trainIdxTable, const NumericTablePtr & testIdxTable)
{
#define DAAL_TRAIN_TEST_SPLIT(cpuId, ...) trainTestSplitImpl<IdxType, cpuId>(__VA_ARGS__);
    DAAL_DISPATCH_FUNCTION_BY_CPU(DAAL_TRAIN_TEST_SPLIT, inputTable, trainTable, testTable, trainIdxTable, testIdxTable);
#undef DAAL_TRAIN_TEST_SPLIT
}

template <typename IdxType>
DAAL_EXPORT void trainTestSplit(const NumericTablePtr & inputTable, const NumericTablePtr & trainTable, const NumericTablePtr & testTable,
                                const NumericTablePtr & trainIdxTable, const NumericTablePtr & testIdxTable)
{
    DAAL_SAFE_CPU_CALL((trainTestSplitDispImpl<IdxType>(inputTable, trainTable, testTable, trainIdxTable, testIdxTable)),
                       (trainTestSplitImpl<IdxType, daal::CpuType::sse2>(inputTable, trainTable, testTable, trainIdxTable, testIdxTable)));
}

template DAAL_EXPORT void trainTestSplit<int>(const NumericTablePtr & inputTable, const NumericTablePtr & trainTable,
                                              const NumericTablePtr & testTable, const NumericTablePtr & trainIdxTable,
                                              const NumericTablePtr & testIdxTable);

} // namespace internal
} // namespace data_management
} // namespace daal
