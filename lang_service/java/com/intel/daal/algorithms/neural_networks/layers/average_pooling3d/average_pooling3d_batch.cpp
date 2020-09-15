/* file: average_pooling3d_batch.cpp */
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

#include <jni.h>
#include "com_intel_daal_algorithms_neural_networks_layers_average_pooling3d_AveragePooling3dBatch.h"

#include "daal.h"

#include "lang_service/java/com/intel/daal/include/common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_average_1pooling3d_AveragePooling3dBatch
 * Method:    cInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_average_1pooling3d_AveragePooling3dBatch_cInit(JNIEnv * env,
                                                                                                                              jobject thisObj,
                                                                                                                              jint prec, jint method,
                                                                                                                              jlong nDim)
{
    return jniBatchLayer<average_pooling3d::Method, average_pooling3d::Batch, average_pooling3d::defaultDense>::newObj(prec, method, nDim);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_average_1pooling3d_AveragePooling3dBatch
 * Method:    cInitParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_average_1pooling3d_AveragePooling3dBatch_cInitParameter(
    JNIEnv * env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatchLayer<average_pooling3d::Method, average_pooling3d::Batch, average_pooling3d::defaultDense>::getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_average_1pooling3d_AveragePooling3dBatch
 * Method:    cGetForwardLayer
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_average_1pooling3d_AveragePooling3dBatch_cGetForwardLayer(
    JNIEnv * env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatchLayer<average_pooling3d::Method, average_pooling3d::Batch, average_pooling3d::defaultDense>::getForwardLayer(prec, method,
                                                                                                                                algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_average_1pooling3d_AveragePooling3dBatch
 * Method:    cGetBackwardLayer
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_average_1pooling3d_AveragePooling3dBatch_cGetBackwardLayer(
    JNIEnv * env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatchLayer<average_pooling3d::Method, average_pooling3d::Batch, average_pooling3d::defaultDense>::getBackwardLayer(prec, method,
                                                                                                                                 algAddr);
}