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
#include "GradientParameter.h"
#include <pthread.h>



struct loop{
	int loopMin;
	int loopMax;
};
struct stData {
	float *deltas;
	float *xList;
	float ySizeF;
	float *yList;
	int layerCount;
	int *neuronCounts;
	float lambda;
	float *thetas;
	int neuronSize;
	int errorSize;
	int deltaSize;
	int xListRows;
	int *dlayerCache;
	int **dMatrixInfo;
	int *nLayerCache;
	int *eLayerCache;
	int numLabels;
	float cost;
	int isLast;
	int loopMin;
	int loopMax;
};
class NeuralNetwork {
private:
	int layerCount;
	int *neuronCounts;
	int numberOfLabels;
	int ySize;
	float ySizefloat;
	float yf;
	float lyf;
	float tyf;
	float lambda;
	int **dMatrixDimensions;
	int *dLayerCache;
	int *nLayerCache;
	int *eLayerCache;
	int dMatrixSize;
	int xDim2;
	int yDim2;
	int neuronSize;
	int errorSize;
	int deltaSize;
	int mDeltaSize;
	float *deltas;
	int xColumns;
	float *xList;
	float *yList;
	int numberOfThreads;
	float **pDeltas;
	struct stData *stDatas;
	pthread_t *threads;
	int threadBarrier;
	struct loop *loops;
public:
	NeuralNetwork(int noThreads, float *alist, float *blist, int layerCount, int *neuronCounts, int numberOfLabels, int ySize, int xColumnSize, float l);
	GradientParameter* calculateBackCostWithThetas(float *thetas);
	static void* calculateBackCost(void *dat);
	float* forwardPropogate(int aListIndex, float *tList, float *xList);
	void predict(float *tList, float *yTemp);
	void predict(int rows, float *xlist, float *tList, float *yTemp);
	float* forwardPropogate(float *tList, float *xList);
	virtual ~NeuralNetwork();
};

#endif /* SRC_NEURALNETWORK_H_ */
