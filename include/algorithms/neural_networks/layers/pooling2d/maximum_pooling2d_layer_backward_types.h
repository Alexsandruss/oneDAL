/* file: maximum_pooling2d_layer_backward_types.h */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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
//  Implementation of backward maximum 2D pooling layer.
//--
*/

#ifndef __MAXIMUM_POOLING2D_LAYER_BACKWARD_TYPES_H__
#define __MAXIMUM_POOLING2D_LAYER_BACKWARD_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_backward_types.h"
#include "algorithms/neural_networks/layers/pooling2d/pooling2d_layer_backward_types.h"
#include "algorithms/neural_networks/layers/pooling2d/maximum_pooling2d_layer_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace maximum_pooling2d
{
/**
 * @defgroup maximum_pooling2d_backward Backward Two-dimensional Max Pooling Layer
 * \copydoc daal::algorithms::neural_networks::layers::maximum_pooling2d::backward
 * @ingroup maximum_pooling2d
 * @{
 */
/**
 * \brief Contains classes for backward maximum 2D pooling layer
 */
namespace backward
{

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__MAXIMUM_POOLING2D__BACKWARD__INPUT"></a>
 * \brief %Input objects for the backward maximum 2D pooling layer
 * \DAAL_DEPRECATED
 */
class DAAL_EXPORT Input : public pooling2d::backward::Input
{
public:
    typedef pooling2d::backward::Input super;
    /**
     * Default constructor
     * \DAAL_DEPRECATED
     */
    Input();

    /**
     * Copy constructor
     * \DAAL_DEPRECATED
     */
    Input(const Input& other);

    /*
     * \DAAL_DEPRECATED
     */
    virtual ~Input() {}

    using layers::backward::Input::get;
    using layers::backward::Input::set;

    /**
     * Returns an input object for backward maximum 2D pooling layer
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     * \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED data_management::TensorPtr get(LayerDataId id) const;

    /**
     * Returns an input object for backward maximum 2D pooling layer
     * \param[in] id    Identifier of the input object
     * \return          %Input object that corresponds to the given identifier
     * \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED data_management::NumericTablePtr get(LayerDataNumericTableId id) const;

    /**
     * Sets an input object for the backward maximum 2D pooling layer
     * \param[in] id  Identifier of the input object
     * \param[in] ptr Pointer to the object
     * \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED void set(LayerDataId id, const data_management::TensorPtr &ptr);

    /**
     * Sets an input object for the backward maximum 2D pooling layer
     * \param[in] id  Identifier of the input object
     * \param[in] ptr Pointer to the object
     * \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED void set(LayerDataNumericTableId id, const data_management::NumericTablePtr &ptr);

    /**
     * Checks an input object for the backward maximum 2D pooling layer
     * \param[in] parameter Algorithm parameter
     * \param[in] method    Computation method
     *
     * \return Status of computations
     * \DAAL_DEPRECATED
     */
    services::Status check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    /*
     * \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED_VIRTUAL virtual data_management::NumericTablePtr getAuxInputDimensions() const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__MAXIMUM_POOLING2D__BACKWARD__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method
 *        of the backward maximum 2D pooling layer
 * \DAAL_DEPRECATED
 */
class DAAL_EXPORT Result : public pooling2d::backward::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result);
    /**
     * Default constructor
     * \DAAL_DEPRECATED
     */
    Result();
    /*
     * \DAAL_DEPRECATED
     */
    virtual ~Result() {}

    using layers::backward::Result::get;
    using layers::backward::Result::set;

    /**
     * Allocates memory to store the result of the backward maximum 2D pooling layer
     * \param[in] input Pointer to an object containing the input data
     * \param[in] parameter %Parameter of the backward maximum 2D pooling layer
     * \param[in] method Computation method for the layer
     *
     * \return Status of computations
     * \DAAL_DEPRECATED
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Checks the result of the backward maximum 2D pooling layer
     * \param[in] input     %Input object for the layer
     * \param[in] parameter %Parameter of the layer
     * \param[in] method    Computation method
     *
     * \return Status of computations
     * \DAAL_DEPRECATED
     */
    DAAL_DEPRECATED services::Status check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    /**
     * \private
     * \DAAL_DEPRECATED
     */
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<Result> ResultPtr;

} // namespace interface1
using interface1::Input;
using interface1::Result;
using interface1::ResultPtr;
} // namespace backward
/** @} */
} // namespace maximum_pooling2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithm
} // namespace daal

#endif