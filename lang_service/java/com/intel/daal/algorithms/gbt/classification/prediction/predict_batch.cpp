/* file: predict_batch.cpp */
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

/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>

#include "daal.h"
#include "com_intel_daal_algorithms_gbt_classification_prediction_PredictionBatch.h"
#include "algorithms/gradient_boosted_trees/gbt_classification_predict_types.h"

#include "lang_service/java/com/intel/daal/include/common_helpers.h"

USING_COMMON_NAMESPACES()
namespace gbtcp = daal::algorithms::gbt::classification::prediction;

/*
* Class:     com_intel_daal_algorithms_gbt_classification_prediction_PredictionBatch
* Method:    cInit
* Signature: (IIJ)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_gbt_classification_prediction_PredictionBatch_cInit(JNIEnv *, jobject thisObj, jint prec,
                                                                                                           jint method, jlong nClasses)
{
    return jniBatch<gbtcp::Method, gbtcp::Batch, gbtcp::defaultDense>::newObj(prec, method, nClasses);
}

/*
* Class:     com_intel_daal_algorithms_gbt_classification_prediction_PredictionBatch
* Method:    cInitParameter
* Signature: (JIII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_gbt_classification_prediction_PredictionBatch_cInitParameter(JNIEnv *, jobject thisObj,
                                                                                                                    jlong algAddr, jint prec,
                                                                                                                    jint method, jint cmode)
{
    return jniBatch<gbtcp::Method, gbtcp::Batch, gbtcp::defaultDense>::getParameter(prec, method, algAddr);
}

/*
* Class:     com_intel_daal_algorithms_gbt_classification_prediction_PredictionBatch
* Method:    cClone
* Signature: (JII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_gbt_classification_prediction_PredictionBatch_cClone(JNIEnv *, jobject thisObj, jlong algAddr,
                                                                                                            jint prec, jint method)
{
    return jniBatch<gbtcp::Method, gbtcp::Batch, gbtcp::defaultDense>::getClone(prec, method, algAddr);
}

/*
* Class:     com_intel_daal_algorithms_gbt_classification_prediction_PredictionParameter
* Method:    cGetNIterations
* Signature: (J)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_gbt_classification_prediction_PredictionParameter_cGetNIterations(JNIEnv * env,
                                                                                                                         jobject thisObj, jlong addr)
{
    return (jlong)(((gbtcp::Parameter *)addr)->nIterations);
}

/*
* Class:     com_intel_daal_algorithms_gbt_classification_prediction_PredictionParameter
* Method:    cSetNIterations
* Signature: (JJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_gbt_classification_prediction_PredictionParameter_cSetNIterations(JNIEnv * env, jobject thisObj,
                                                                                                                        jlong addr, jlong value)
{
    ((gbtcp::Parameter *)addr)->nIterations = value;
}