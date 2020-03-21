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

#include "NeuralNetwork.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits>
#include "avx2.h"

#include <sys/time.h>
using namespace std;

typedef std::numeric_limits<float> lim_dbl;
static const float LOWER_BOUND = (lim_dbl::epsilon() / lim_dbl::radix);
static const float UPPER_BOUND = 1.0f - (LOWER_BOUND * 2);
static const __m256 ones = _mm256_set1_ps(1);
static const __m256 zeros = _mm256_set1_ps(0);
#define E exp(1.0)
NeuralNetwork::NeuralNetwork(int noThreads, float *alist, float *blist, int lCount, int *nCounts, int nOfLabels, int yWeight, int xColumnSize, float l) {

	deltas = 0;
	lambda = l;
	numberOfThreads = noThreads;
	threadLast = numberOfThreads - 1;
	xList = alist;
	yList = blist;
	xColumns = xColumnSize;
	layerCount = lCount;
	neuronCounts = nCounts;
	numberOfLabels = nOfLabels;
	ySize = yWeight;
	dMatrixDimensions = new int*[layerCount - 1];
	dLayerCache = new int[layerCount];
	nLayerCache = new int[layerCount + 1];
	eLayerCache = new int[layerCount];
	dMatrixSize = 0;
	xDim2 = neuronCounts[0];
	yDim2 = numberOfLabels;
	nLayerCache[0] = 0;
	eLayerCache[0] = 0;
	dLayerCache[0] = 0;
	deltaSize = 0;
	neuronSize = 0;
	errorSize = 0;
	ySizefloat = ySize;
	yf = 1.0f / ySizefloat;
	lyf = lambda / ySizefloat;
	tyf = lambda / (2.0f * ySizefloat);
	pDeltas = (float**) malloc(sizeof(float*) * numberOfThreads);
	stDatas = (struct stData*) malloc(sizeof(struct stData) * numberOfThreads);
	threads = (pthread_t*) malloc(sizeof(pthread_t) * threadLast);
	loops = (struct loop*) malloc(sizeof(struct loop) * numberOfThreads);
	for (int t = 0; t < numberOfThreads; ++t) {
		loops[t].loopMin = (int) ((long) (t + 0) * (long) (ySize) / (long) numberOfThreads);
		loops[t].loopMax = (int) ((long) (t + 1) * (long) (ySize) / (long) numberOfThreads);
	}
	//we need rowcount in float value for calculation

	for (int i = 0; i < layerCount; ++i) {

		neuronSize += i == layerCount - 1 ? neuronCounts[i] : neuronCounts[i] + 1;
		nLayerCache[i + 1] = neuronSize;

		if (i < layerCount - 1) {

			errorSize += i == layerCount - 2 ? neuronCounts[i + 1] : neuronCounts[i + 1] + 1;
			eLayerCache[i + 1] = errorSize;
			dMatrixDimensions[i] = new int[2];
			dMatrixDimensions[i][0] = neuronCounts[i + 1];
			dMatrixDimensions[i][1] = neuronCounts[i] + 1;

			deltaSize += (dMatrixDimensions[i][0] * dMatrixDimensions[i][1]);
			dLayerCache[i + 1] = deltaSize;
		}
	}

	mDeltaSize = sizeof(float) * deltaSize;

}

NeuralNetwork::~NeuralNetwork() {
	delete[] dLayerCache;
	delete[] nLayerCache;
	delete[] eLayerCache;
	free(xList);
	delete[] yList;
	free(stDatas);
	free(pDeltas);
	free(threads);

}

void _mulAdd(float *a, float *b, float *s) {
	__m256 va = _mm256_loadu_ps(a);
	__m256 vb = _mm256_loadu_ps(b);
	__m256 vc = _mm256_mul_ps(va, vb);
	for (int i = 0; i < 8; i++) {
		s[0] += vc[i];
	}

}

void _mulAddBroadcast(float *d, float *e, float *n) {
	__m256 vd = _mm256_loadu_ps(d);
	__m256 vn = _mm256_loadu_ps(n);
	__m256 ve = _mm256_broadcast_ss(e);

	__m256 res = _mm256_fmadd_ps(ve, vn, vd);
	_mm256_storeu_ps(d, res);
}

void _sums(float *ylist, float *neurons, float *sum) {

	__m256 y = _mm256_loadu_ps(ylist);
	__m256 n = _mm256_loadu_ps(neurons);
	__m256 negateY = _mm256_sub_ps(zeros, y);
	__m256 minusY = _mm256_sub_ps(ones, y);
	__m256 minusN = _mm256_sub_ps(ones, n);
	__m256 res = _mm256_sub_ps(_mm256_mul_ps(negateY, log256_ps(n)), _mm256_mul_ps(minusY, log256_ps(minusN)));
	for (int i = 0; i < 8; i++) {
		sum[0] += res[i];
	}
}

void* NeuralNetwork::calculateBackCost(void *dat) {
	struct stData *data = (struct stData*) dat;
	data->cost = 0;
	float *neurons = (float*) malloc(sizeof(float) * data->neuronSize);
	float *errors = (float*) malloc(sizeof(float) * data->errorSize);
	data->deltas = (float*) malloc(sizeof(float) * data->deltaSize);
	for (int i = 0; i < data->deltaSize; ++i) {
		data->deltas[i] = 0;
	}
	int layerCount = data->layerCount;
	int *neuronCounts = data->neuronCounts;
	float *thetas = data->thetas;
	float *xList = data->xList;
	float *yList = data->yList;
	int *dlayerCache = data->dlayerCache;
	int numLabels = data->numLabels;
	int xListRows = data->xListRows;
	int **dMatrixInfo = data->dMatrixInfo;
	int *nLayerCache = data->nLayerCache;
	int *eLayerCache = data->eLayerCache;

	for (int m = data->loopMin; m < data->loopMax; m++) {
		int yCache = m * numLabels;
		int xCache = xListRows * m;
		float *x = &(xList[xCache]);
		float *y = &(yList[yCache]);

		//forward propagate
		for (int l = 0; l < layerCount; l++) {
			int lPrev = l - 1;
			int previousLayer = nLayerCache[l];
			bool isLast = l == (layerCount - 1);
			int neuronSize = isLast ? neuronCounts[l] : neuronCounts[l] + 1;
			for (int j = 0; j < neuronSize; j++) {
				int jPrev = j - 1;
				int row = previousLayer + j;
				neurons[row] = .0f;

				if (j == 0 && !isLast) {
					neurons[row] = 1;
				} else if (l == 0) {
					neurons[row] = x[jPrev];
				} else {
					float *t = &(thetas[(dMatrixInfo[lPrev][1] * (isLast ? j : jPrev)) + dlayerCache[lPrev]]);
					float *n = &(neurons[nLayerCache[lPrev]]);
					int nCounts = neuronCounts[lPrev] + 1;
					int siz = nCounts - (nCounts & 7);
					for (int k = 0; k < siz; k = k + 8) {
						_mulAdd(&t[k], &n[k], &neurons[row]);
					}
					for (int k = siz; k < nCounts; k++) {
						neurons[row] += t[k] * n[k];
					}

					neurons[row] = (UPPER_BOUND / (1 + pow(E, -neurons[row]))) + LOWER_BOUND;
				}
			}
		}

		//backpropagate
		for (int i = layerCount - 2; i >= 0; i--) {
			int iNext = i + 1;
			int neuronSize = i == layerCount - 2 ? neuronCounts[iNext] : neuronCounts[iNext] + 1;
			int previousLayer = eLayerCache[i];
			int nCache = nLayerCache[iNext];

			int dCache = dlayerCache[iNext];
			int eCache = eLayerCache[iNext];
			float *e = &(errors[eCache]);
			float *t = &(thetas[dCache]);
			for (int j = neuronSize - 1; j >= 0; j--) {
				int row = previousLayer + j;

				errors[row] = 0; //reset
				float nVal = neurons[nCache + j];
				if (i == layerCount - 2) {
					errors[row] = nVal - y[j];
				} else {
					int nCounts = neuronCounts[i + 2];
					int isLast = nCounts - 1;
					float sigmoid = (nVal * (1 - nVal));
					float *t2 = &(t[j]);
					int dif = nCounts & 3;
					int siz = nCounts - dif;
					int val = dMatrixInfo[iNext][1];
					for (int k = 0; k < siz; k = k + 4) {
						errors[row] += t2[val * k] * e[k];
						errors[row] += t2[val * (k + 1)] * e[k + 1];
						errors[row] += t2[val * (k + 2)] * e[k + 2];
						errors[row] += t2[val * (k + 3)] * e[k + 3];

					}

					for (int a = siz; a < nCounts; a++) {
						errors[row] += t2[val * a] * e[a];

						if (a == isLast) {
							errors[row] = errors[row] * sigmoid;
						}
					}
				}

			}
		}

		//calculate deltas
		float sum = 0.0;
		for (int i = 0; i < layerCount - 1; i++) {
			int iNext = i + 1;
			int n1 = neuronCounts[iNext];
			int n2 = neuronCounts[i] + 1;
			int nCache1 = nLayerCache[iNext];
			int eCache = eLayerCache[i];
			int nCache = nLayerCache[i];
			float *e = &(errors[eCache]);
			float *n = &(neurons[nCache]);
			int isLast = i == layerCount - 2;
			int dCache = dlayerCache[i];
			float *d = &(data->deltas[dCache]);
			int dif = n1 & 7;
			int siz = n1 - dif;
			for (int j = 0; j < siz; j = j + 8) {
				if (isLast) {
					_sums(&yList[yCache + j], &neurons[nCache1 + j], &sum);
				}
				int index = i == 0 ? j + 1 : j;
				int m = j;
				float eVal = e[index++];
				float eVal2 = e[index++];
				float eVal3 = e[index++];
				float eVal4 = e[index++];
				float eVal5 = e[index++];
				float eVal6 = e[index++];
				float eVal7 = e[index++];
				float eVal8 = e[index++];
				float *d2 = &(d[dMatrixInfo[i][1] * m++]);
				float *d22 = &(d[dMatrixInfo[i][1] * m++]);
				float *d23 = &(d[dMatrixInfo[i][1] * m++]);
				float *d24 = &(d[dMatrixInfo[i][1] * m++]);
				float *d25 = &(d[dMatrixInfo[i][1] * m++]);
				float *d26 = &(d[dMatrixInfo[i][1] * m++]);
				float *d27 = &(d[dMatrixInfo[i][1] * m++]);
				float *d28 = &(d[dMatrixInfo[i][1] * m++]);
				int diff = n2 & 7;
				int size = n2 - diff;
				for (int k = 0; k < size; k = k + 8) {
					_mulAddBroadcast(&d2[k], &eVal, &n[k]);
					_mulAddBroadcast(&d22[k], &eVal2, &n[k]);
					_mulAddBroadcast(&d23[k], &eVal3, &n[k]);
					_mulAddBroadcast(&d24[k], &eVal4, &n[k]);
					_mulAddBroadcast(&d25[k], &eVal5, &n[k]);
					_mulAddBroadcast(&d26[k], &eVal6, &n[k]);
					_mulAddBroadcast(&d27[k], &eVal7, &n[k]);
					_mulAddBroadcast(&d28[k], &eVal8, &n[k]);
				}
				for (int d = 0; d < diff; d++) {
					float nVal = n[size + d];
					d2[size + d] += eVal * nVal;
					d22[size + d] += eVal2 * nVal;
					d23[size + d] += eVal3 * nVal;
					d24[size + d] += eVal4 * nVal;
					d25[size + d] += eVal5 * nVal;
					d26[size + d] += eVal6 * nVal;
					d27[size + d] += eVal7 * nVal;
					d28[size + d] += eVal8 * nVal;
				}
			}

			for (int a = 0; a < dif; a++) {
				int j = a + siz;
				if (isLast) {

					sum += ((-1 * yList[yCache + j]) * log(neurons[nCache1 + j])) - ((1 - yList[yCache + j]) * log(1 - neurons[nCache1 + j]));
				}
				int index = i == 0 ? j + 1 : j;
				int drcache = (dMatrixInfo[i][1] * j);
				float eVal = e[index];
				float *d2 = &(d[drcache]);
				int diff = n2 & 7;
				int size = n2 - diff;
				for (int k = 0; k < size; k = k + 8) {
					_mulAddBroadcast(&d2[k], &eVal, &n[k]);
				}
				for (int d = 0; d < diff; d++) {
					float nVal = n[size + d];
					d2[size + d] += eVal * nVal;
				}
			}

		}
		data->cost += data->ySizeF * sum;

	}

	free(neurons);
	free(errors);

	if (!data->isLast) {
		pthread_exit(NULL);
	}

	return 0;
}

GradientParameter* NeuralNetwork::calculateBackCostWithThetas(float *thetas) {
//allocate place for deltas
	deltas = (float*) malloc(mDeltaSize);

//create params for each thread

	float cost = 0.0f;
	for (int t = 0; t < numberOfThreads; ++t) {
		int isLast = t == (numberOfThreads - 1);
		stDatas[t].deltas = &(pDeltas[t][0]);
		stDatas[t].xList = xList;
		stDatas[t].ySizeF = yf;
		stDatas[t].yList = yList;
		stDatas[t].layerCount = layerCount;
		stDatas[t].neuronCounts = neuronCounts;
		stDatas[t].lambda = lambda;
		stDatas[t].thetas = thetas;
		stDatas[t].neuronSize = neuronSize;
		stDatas[t].errorSize = errorSize;
		stDatas[t].deltaSize = deltaSize;
		stDatas[t].xListRows = xColumns;
		stDatas[t].dlayerCache = dLayerCache;
		stDatas[t].dMatrixInfo = dMatrixDimensions;
		stDatas[t].nLayerCache = nLayerCache;
		stDatas[t].eLayerCache = eLayerCache;
		stDatas[t].numLabels = numberOfLabels;
		stDatas[t].cost = 0.0f;
		stDatas[t].isLast = isLast;
		stDatas[t].loopMin = loops[t].loopMin;
		stDatas[t].loopMax = loops[t].loopMax;

		if (!isLast) {
			pthread_create(&threads[t], NULL, calculateBackCost, (void*) &(stDatas[t]));

		} else {
			//if its last handle by main thread
			this->calculateBackCost(&stDatas[t]);
		}

	}

//wait for other threads
	for (int t = 0; t < numberOfThreads - 1; t++) {
		pthread_join(threads[t], NULL);
	}

	float thetaSum = 0.0;

//collect all data from threads and update cost
	int da = 0;
	for (int l = 0; l < deltaSize; l++) {
		int dc = (l - dLayerCache[da]) % dMatrixDimensions[da][1];
		deltas[l] = 0.0;
		for (int i = 0; i < numberOfThreads; i++) {

			deltas[l] += stDatas[i].deltas[l];

		}
		deltas[l] *= yf;
		if (dc > 0) {
			deltas[l] += lyf * thetas[l];
			thetaSum += pow(thetas[l], 2);
		}

		if ((l + 1) == dLayerCache[da + 1]) {
			da++;
		}
	}

	for (int i = 0; i < numberOfThreads; ++i) {
		cost += stDatas[i].cost;
		free(stDatas[i].deltas);
	}

	cost += thetaSum * tyf;

	return new GradientParameter(deltas, cost);;

}

void NeuralNetwork::predict(float *tList, float *yTemp) {

	int totalCorrect = 0;
	int totalWrong = 0;

	for (int i = 0; i < ySize; ++i) {

		float *neurons = forwardPropogate(i, tList, &(xList[(i * xColumns)]));
		float closer = RAND_MAX;
		float val = 0;
		for (int j = 0; j < numberOfLabels; j++) {

			if (fabs((1 - closer)) > fabs((1 - neurons[nLayerCache[layerCount - 1] + j]))) {
				val = j + 1;
				closer = neurons[nLayerCache[layerCount - 1] + j];
			}
		}

		if (yTemp[i] == val) {
			totalCorrect++;
		} else {
			totalWrong++;
		}

		free(neurons);

	}

	printf("\nPrediction complete. Total %i correct and %i wrong prediction\n", totalCorrect, totalWrong);
	float successRate = totalCorrect * 100 / ySize;
	printf("\n Success rate is: %%%0.0f\n", successRate);
}

void NeuralNetwork::predict(int rows, float *xlist, float *tList, float *yTemp) {

	int totalCorrect = 0;
	int totalWrong = 0;

	for (int i = 0; i < rows; ++i) {

		float *neurons = forwardPropogate(tList, &(xlist[(i * xColumns)]));
		float closer = RAND_MAX;
		float val = 0;
		for (int j = 0; j < numberOfLabels; j++) {

			if (fabs((1 - closer)) > fabs((1 - neurons[nLayerCache[layerCount - 1] + j]))) {
				val = j + 1;
				closer = neurons[nLayerCache[layerCount - 1] + j];
			}
		}

		if (yTemp[i] == val) {
			totalCorrect++;
		} else {
			totalWrong++;
		}

		free(neurons);

	}
	float successRate = totalCorrect * 100 / rows;
	printf("\n\t|\n\t\\__Prediction complete. Total %i correct and %i wrong prediction, rate: %%%0.0f \n", totalCorrect, totalWrong, successRate);

}
float* NeuralNetwork::forwardPropogate(int aListIndex, float *tList, float *xList) {

	int mNeuronSize = sizeof(float) * neuronSize;
	float *neurons = (float*) malloc(mNeuronSize);

	for (int l = 0; l < layerCount; l++) {
		int lPrev = l - 1;
		int previousLayer = nLayerCache[l];
		bool isLast = l == layerCount - 1;
		int neuronSize = isLast ? neuronCounts[l] : neuronCounts[l] + 1;
		for (int j = 0; j < neuronSize; j++) {
			int jPrev = j - 1;
			int row = previousLayer + j;
			neurons[row] = 0;

			if (j == 0 && !isLast) {
				neurons[row] = 1;
			} else if (l == 0) {
				neurons[row] = xList[jPrev];
			} else {
				float *t = &(tList[(dMatrixDimensions[lPrev][1] * (isLast ? j : jPrev)) + dLayerCache[lPrev]]);
				float *n = &(neurons[nLayerCache[lPrev]]);
				int nCounts = neuronCounts[lPrev] + 1;
				int dif = nCounts & 7;
				int siz = nCounts - dif;
				for (int k = 0; k < siz; k = k + 8) {
					_mulAdd(&t[k], &n[k], &neurons[row]);
				}
				for (int k = siz; k < nCounts; k++) {
					neurons[row] += t[k] * n[k];
				}

				neurons[row] = (UPPER_BOUND / (UPPER_BOUND + pow(E, -neurons[row])));
			}
		}
	}

	return neurons;
}
float* NeuralNetwork::forwardPropogate(float *tList, float *xList) {

	int mNeuronSize = sizeof(float) * neuronSize;
	float *neurons = (float*) malloc(mNeuronSize);
	for (int l = 0; l < layerCount; l++) {
		int lPrev = l - 1;
		int previousLayer = nLayerCache[l];
		bool isLast = l == layerCount - 1;
		int neuronSize = isLast ? neuronCounts[l] : neuronCounts[l] + 1;
		for (int j = 0; j < neuronSize; j++) {
			int jPrev = j - 1;
			int row = previousLayer + j;
			neurons[row] = 0;

			if (j == 0 && !isLast) {
				neurons[row] = 1;
			} else if (l == 0) {
				neurons[row] = xList[jPrev];
			} else {
				float *t = &(tList[(dMatrixDimensions[lPrev][1] * (isLast ? j : jPrev)) + dLayerCache[lPrev]]);
				float *n = &(neurons[nLayerCache[lPrev]]);
				int nCounts = neuronCounts[lPrev] + 1;
				int dif = nCounts & 7;
				int siz = nCounts - dif;
				for (int k = 0; k < siz; k = k + 8) {
					_mulAdd(&t[k], &n[k], &neurons[row]);
				}
				for (int k = siz; k < nCounts; k++) {
					neurons[row] += t[k] * n[k];
				}

				neurons[row] = (UPPER_BOUND / (UPPER_BOUND + pow(E, -neurons[row])));
			}
		}
	}

	return neurons;
}
