/*
 Copyright (c) 2015, Burak Sarac, burak@linux.com
 All rights reserved.
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that
 the following conditions are met:
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
 following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
 following disclaimer in the documentation and/or other materials provided with the distribution.
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
 EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
 THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
 OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
 TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef SRC_NEURALNETWORK_H_
#define SRC_NEURALNETWORK_H_
#include "ApplicationParameters.h"
#include <pthread.h>

struct loop {
	llu loopMin;
	llu loopMax;
};
struct stData {
	float *deltas;
	float *xList;
	float ySizeF;
	float *yList;
	llu layerCount;
	llu *neuronCounts;
	float lambda;
	float *thetas;
	llu neuronSize;
	llu errorSize;
	llu deltaSize;
	llu xListRows;
	llu *dlayerCache;
	llu **dMatrixInfo;
	llu *nLayerCache;
	llu *eLayerCache;
	llu numLabels;
	float cost;
	llu isLast;
	llu loopMin;
	llu loopMax;
	pthread_cond_t waitCond;
	pthread_cond_t completeCond;
	pthread_mutex_t mutex;
	llu workType;
	//fmin params
	float *x;
	float *x0;
	float *df1;
	float *df0;
	float z1;
	float *s;
	float *df2;
	float d1;
	float d2;
	float z2;
	float sum1;
	float sum2;
	float sum3;
	float p;
	llu size;
	llu end;
	llu isMain;
	float* calculatedDeltas;
	llu tloopmin;
};
class NeuralNetwork {
private:
	llu layerCount;
	llu *neuronCounts;
	llu numberOfLabels;
	llu ySize;
	float ySizefloat;
	float yf;
	float lyf;
	float tyf;
	llu **dMatrixDimensions;
	llu *dLayerCache;
	llu *nLayerCache;
	llu *eLayerCache;
	llu dMatrixSize;
	llu neuronSize;
	llu errorSize;
	llu deltaSize;
	llu mDeltaSize;

	llu xColumns;
	float *xList;
	float *yList;
	unsigned long int numberOfThreads;

	pthread_t *threads;
	llu threadBarrier;
public:
	struct stData *stDatas;
	float  *deltas;
	NeuralNetwork(ApplicationParameters *params, float *alist, float *blist, llu *neuronCounts);
	float calculateBackCostWithThetas(float *thetas);
	static inline  void* calculateBackCost(void *dat);
	static inline  void calculateCost(struct stData *data);
	float* forwardPropogate(llu aListIndex, float *tList, float *xList);
	void predict(float *tList, float *yTemp);
	void predict(llu rows, float *xlist, float *tList, float *yTemp);
	void submitWork(llu workType);
	float* forwardPropogate(float *tList, float *xList);
	static inline  void handleWork(stData *param);
	virtual ~NeuralNetwork();
};

#endif /* SRC_NEURALNETWORK_H_ */
