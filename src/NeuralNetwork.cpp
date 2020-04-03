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
#include "x86intrin.h"
#include <thread>
#include <sys/time.h>
#include <xmmintrin.h>
using namespace std;

typedef std::numeric_limits<float> lim_dbl;
static const float LOWER_BOUND = (lim_dbl::epsilon() / lim_dbl::radix);
static const float UPPER_BOUND = 1.0f - (LOWER_BOUND * 2);
static const __m256 ones = _mm256_set1_ps(1);
static const __m256 zeros = _mm256_set1_ps(0);
static const __m256 V_ALL_SET = _mm256_set1_ps(-1);
#define E exp(1.0)
NeuralNetwork::NeuralNetwork(ApplicationParameters *params, float *alist, float *blist, int *nCounts) {
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_mm_setcsr(_mm_getcsr() | 0x8040);
	numberOfThreads = params->getNumberOfThreads();
	xList = alist;
	yList = blist;
	xColumns = params->getColumnCount();
	layerCount = params->getTotalLayerCount();
	neuronCounts = nCounts;
	numberOfLabels = params->getNumberOfLabels();
	ySize = params->getRowCount();
	dMatrixDimensions = new int*[layerCount - 1];
	dLayerCache = new int[layerCount];
	nLayerCache = new int[layerCount + 1];
	eLayerCache = new int[layerCount];
	dMatrixSize = 0;
	nLayerCache[0] = 0;
	eLayerCache[0] = 0;
	dLayerCache[0] = 0;
	deltaSize = 0;
	neuronSize = 0;
	errorSize = 0;
	ySizefloat = ySize;
	yf = 1 / ySizefloat;
	lyf = params->getLambda() / ySizefloat;
	tyf = params->getLambda() / (2.0f * ySizefloat);

	stDatas = (struct stData*) malloc(sizeof(struct stData) * numberOfThreads);
	threads = (pthread_t*) malloc(sizeof(pthread_t) * numberOfThreads - 1);
	threadBarrier = params->getCpus() - numberOfThreads;

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
	deltas = (float*) malloc(sizeof(float) * 1);
	for (int i = params->getCpus() - 1; i >= threadBarrier; i--) {
		int t = i - threadBarrier;
		int isMain = t == 0;
		int loopmin = (int) ((long) (t + 0) * (long) (ySize) / (long) numberOfThreads);
		int loopmax = (int) ((long) (t + 1) * (long) (ySize) / (long) numberOfThreads);
		stDatas[t].deltas = (float*) malloc(sizeof(float) * deltaSize);
		stDatas[t].xList = xList;
		stDatas[t].ySizeF = yf;
		stDatas[t].yList = yList;
		stDatas[t].layerCount = layerCount;
		stDatas[t].neuronCounts = neuronCounts;
		stDatas[t].lambda = params->getLambda();
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
		stDatas[t].isLast = isMain;
		stDatas[t].loopMin = loopmin;
		stDatas[t].loopMax = loopmax;
		stDatas[t].workType = 0;

		if (!stDatas[t].isLast) {
			stDatas[t].mutex = PTHREAD_MUTEX_INITIALIZER;
			stDatas[t].completeCond = PTHREAD_COND_INITIALIZER;
			stDatas[t].waitCond = PTHREAD_COND_INITIALIZER;
			pthread_create(&threads[t], NULL, calculateBackCost, (void*) &(stDatas[t]));
			cpu_set_t cpuset;
			CPU_ZERO(&cpuset);
			CPU_SET(i, &cpuset);
			pthread_setaffinity_np(threads[t], sizeof(cpu_set_t), &cpuset);

		}

	}

}

NeuralNetwork::~NeuralNetwork() {
	for (int t = numberOfThreads - 1; t >= 0; t--) {
		if (!stDatas[t].isLast) {
			pthread_mutex_lock(&stDatas[t].mutex);
			stDatas[t].workType = -1;
			pthread_mutex_unlock(&stDatas[t].mutex);
			pthread_cond_signal(&stDatas[t].waitCond);
		}

	}
	for (int t = numberOfThreads - 1; t > 0; t--) {
		pthread_join(threads[t], NULL);
		pthread_mutex_destroy(&stDatas[t].mutex);
		pthread_cond_destroy(&stDatas[t].waitCond);
		pthread_cond_destroy(&stDatas[t].completeCond);
	}
	delete[] dLayerCache;
	delete[] nLayerCache;
	delete[] eLayerCache;
	free(xList);
	delete[] yList;

	for (int i = 0; i < numberOfThreads; ++i) {
		free(stDatas[i].deltas);
	}
	free(stDatas);
	free(threads);
	free(deltas);

}

static inline float _mulAdd(float *a, float *b) {
	__m256 va = _mm256_loadu_ps(a);
	__m256 vb = _mm256_loadu_ps(b);
	return sum8(_mm256_mul_ps(va, vb));
}

static inline void _mulAddBroadcast(float *d, float *e, float *n) {
	__m256 vd = _mm256_loadu_ps(d);
	__m256 vn = _mm256_loadu_ps(n);
	__m256 ve = _mm256_broadcast_ss(e);
	_mm256_storeu_ps(d, _mm256_macc_ps(ve, vn, vd));
}

static inline float _sums(float *ylist, float *neurons) {

	__m256 y = _mm256_loadu_ps(ylist);
	__m256 n = _mm256_loadu_ps(neurons);
	__m256 minusY = _mm256_sub_ps(ones, y);
	__m256 minusN = _mm256_sub_ps(ones, n);
	__m256 res = _mm256_nmsub_ps(y, log256_ps(n), _mm256_mul_ps(minusY, log256_ps(minusN)));
	return sum8(res);
}

static inline float _mulFmaddStore1(float *a) {
	__m256 va = _mm256_loadu_ps(a);
	__m256 vres = _mm256_mul_ps(va, va);
	return sum8(_mm256_mul_ps(V_ALL_SET, vres));
}
static inline float _mulFmaddStore2(float *a, float *b) {
	__m256 va = _mm256_loadu_ps(a);
	__m256 vb = _mm256_loadu_ps(b);
	return sum8(_mm256_mul_ps(va, vb));
}
static inline void _mulNegate(float *a, float *d) {
	__m256 va = _mm256_loadu_ps(a);
	__m256 vres = _mm256_mul_ps(va, V_ALL_SET);
	_mm256_storeu_ps(d, vres);
}
static inline void _mcopy(float *a, float *b) {
	__m256 buffer = _mm256_loadu_ps(a);
	_mm256_storeu_ps(b, buffer);
}

static inline void _mulSub(float *a, float *b, float *c) {
	__m256 va = _mm256_broadcast_ss(a);
	__m256 vb = _mm256_loadu_ps(b);
	__m256 vc = _mm256_loadu_ps(c);
	_mm256_storeu_ps(b, _mm256_msub_ps(va, vb, vc));
}
static inline void _mswap(float *a, float *b) {
	__m256 va = _mm256_loadu_ps(a);
	__m256 vb = _mm256_loadu_ps(b);
	_mm256_storeu_ps(a, vb);
	_mm256_storeu_ps(b, va);
}

static inline void fWork1(stData *param) {

	for (int r = 0; r < param->size; r += 8) {
		_mcopy(&param->x[r], &param->x0[r]);
		_mcopy(&param->df1[r], &param->df0[r]);
		_mulAddBroadcast(&param->x[r], &param->z1, &param->s[r]);
	}

	for (int r = param->size; r < param->end; r++) {
		param->x0[r] = param->x[r]; //copy x value into x0
		param->df0[r] = param->df1[r]; //copy df1 value into df0
		param->x[r] = fma(param->z1, param->s[r], param->x[r]); //update x as X = X + z1*s;
	}

}

static inline void fWork2(stData *param) {
	param->d2 = 0.0;
	for (int r = 0; r < param->size; r += 8) {
		_mcopy(&(param->calculatedDeltas[r]), &param->df2[r]);
		param->d2 += _mulFmaddStore2(&param->s[r], &param->df2[r]);
	}
	for (int r = param->size; r < param->end; r++) {
		param->df2[r] = param->calculatedDeltas[r];
		param->d2 = fma(param->s[r], param->df2[r], param->d2); // d2 = df2'*s;
	}
}

static inline void fWork3(stData *param) {
	for (int r = 0; r < param->size; r += 8) {
		_mulAddBroadcast(&param->x[r], &param->z2, &param->s[r]);
	}
	for (int r = param->size; r < param->end; r++) {
		param->x[r] = fma(param->z2, param->s[r], param->x[r]);
	}
}

static inline void fWork4(stData *param) {
	param->d2 = 0.0;
	for (int r = 0; r < param->size; r += 8) {
		_mcopy(&(param->calculatedDeltas[r]), &param->df2[r]);
		param->d2 += _mulFmaddStore2(&param->s[r], &param->df2[r]);
	}
	for (int r = param->size; r < param->end; r++) {
		param->df2[r] = param->calculatedDeltas[r];
		param->d2 = fma(param->s[r], param->df2[r], param->d2); // d2 = df2'*s;
	}
}

static inline void fWork5(stData *param) {
	for (int r = 0; r < param->size; r += 8) {
		_mulAddBroadcast(&param->x[r], &param->z2, &param->s[r]);
	}
	for (int r = param->size; r < param->end; r++) {
		param->x[r] = fma(param->z2, param->s[r], param->x[r]);
	}
}

static inline void fWork6(stData *param) {
	param->d2 = 0.0;
	for (int r = 0; r < param->size; r += 8) {
		_mcopy(&(param->calculatedDeltas[r]), &param->df2[r]);
		param->d2 += _mulFmaddStore2(&param->s[r], &param->df2[r]);
	}
	for (int r = param->size; r < param->end; r++) {
		param->df2[r] = param->calculatedDeltas[r];
		param->d2 = fma(param->s[r], param->df2[r], param->d2); // d2 = df2'*s;
	}
}

static inline void fWork7(stData *param) {
	param->sum1 = 0.0;
	param->sum2 = 0.0;
	param->sum3 = 0.0;
	for (int r = 0; r < param->size; r += 8) {
		param->sum1 += _mulFmaddStore2(&param->df2[r], &param->df2[r]);
		param->sum2 += _mulFmaddStore2(&param->df1[r], &param->df2[r]);
		param->sum3 += _mulFmaddStore2(&param->df1[r], &param->df1[r]);
	}
	for (int r = param->size; r < param->end; r++) {
		param->sum1 = fma(param->df2[r], param->df2[r], param->sum1);
		param->sum2 = fma(param->df1[r], param->df2[r], param->sum2);
		param->sum3 = fma(param->df1[r], param->df1[r], param->sum3);
	}
}

static inline void fWork8(stData *param) {
	param->d2 = 0.0;
	for (int r = 0; r < param->size; r += 8) {
		_mulSub(&param->p, &param->s[r], &param->df2[r]);
		_mswap(&param->df1[r], &param->df2[r]);
		param->d2 += _mulFmaddStore2(&param->s[r], &param->df1[r]);
	}
	for (int r = param->size; r < param->end; r++) {
		param->s[r] = param->p * param->s[r] - param->df2[r];
		float tmp = param->df1[r];
		param->df1[r] = param->df2[r];
		param->df2[r] = tmp;
		param->d2 = fma(param->df1[r], param->s[r], param->d2); // d2 = df1'*s;
	}
}

static inline void fWork9(stData *param) {
	param->d2 = 0.0;
	for (int r = 0; r < param->size; r += 8) {
		_mulNegate(&param->df1[r], &param->s[r]);
		param->d2 += _mulFmaddStore1(&param->s[r]);
	}
	for (int r = param->size; r < param->end; r++) {
		param->s[r] = -param->df1[r]; // s = -df1;
		param->d2 = fma(-param->s[r], param->s[r], param->d2);
	}
}

static inline void fWork10(stData *param) {
	for (int r = 0; r < param->size; r += 8) {
		_mcopy(&param->x0[r], &param->x[r]);
		_mcopy(&param->df0[r], &param->df1[r]);
	}
	for (int r = param->size; r < param->end; r++) {
		param->x[r] = param->x0[r];
		param->df1[r] = param->df0[r];
	}
}

static inline void fWork11(stData *param) {
	param->d1 = 0.0;
	for (int r = 0; r < param->size; r += 8) {
		_mswap(&param->df1[r], &param->df2[r]);
		_mulNegate(&param->df1[r], &param->s[r]);
		param->d1 += _mulFmaddStore1(&param->s[r]);
	}
	for (int r = param->size; r < param->end; r++) {
		float tmp = param->df1[r];
		param->df1[r] = param->df2[r];
		param->df2[r] = tmp;
		param->s[r] = -param->df1[r];
		param->d1 = fma(-param->s[r], param->s[r], param->d1);
	}
}

static inline void fWork13(stData *param) {
	param->d1 = 0.0;
	for (int r = 0; r < param->size; r += 8) {
		_mcopy(&(param->calculatedDeltas[r]), &(param->df1[r]));
		_mulNegate(&param->df1[r], &param->s[r]);
		param->d1 += _mulFmaddStore1(&param->s[r]);
	}
	for (int r = param->size; r < param->end; r++) {
		param->df1[r] = param->calculatedDeltas[r];
		param->s[r] = -param->df1[r];
		param->d1 = fma(-param->s[r], param->s[r], param->d1);
	}
}

void NeuralNetwork::calculateCost(struct stData *data) {
	float *neurons = (float*) malloc(sizeof(float) * data->neuronSize);
	float *errors = (float*) malloc(sizeof(float) * data->errorSize);
	data->cost = 0.0f;
	int dSize = data->deltaSize - (data->deltaSize & 7);
	for (int i = 0; i < dSize; i += 8) {
		_mm256_storeu_ps(&(data->deltas[i]), zeros);
	}
	for (int i = dSize; i < data->deltaSize; i++) {
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
					int lPrev = l - 1;
					int dCache = dlayerCache[l - 1];
					int nCounts = neuronCounts[lPrev] + 1;
					int siz = nCounts - (nCounts & 7);
					float *n = &(neurons[nLayerCache[lPrev]]);
					float *t = &(thetas[(dMatrixInfo[lPrev][1] * (isLast ? j : jPrev)) + dCache]);
					for (int k = 0; k < siz; k = k + 8) {
						neurons[row] += _mulAdd(&t[k], &n[k]);
					}
					for (int k = siz; k < nCounts; k++) {
						neurons[row] = fma(t[k], n[k], neurons[row]);
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
					int siz = nCounts - (nCounts & 3);
					int val = dMatrixInfo[iNext][1];
					for (int k = 0; k < siz; k = k + 4) {
						errors[row] = fma(t2[val * k], e[k], errors[row]);
						errors[row] = fma(t2[val * (k + 1)], e[k + 1], errors[row]);
						errors[row] = fma(t2[val * (k + 2)], e[k + 2], errors[row]);
						errors[row] = fma(t2[val * (k + 3)], e[k + 3], errors[row]);

					}

					for (int a = siz; a < nCounts; a++) {
						errors[row] = fma(t2[val * a], e[a], errors[row]);

						errors[row] = a == isLast ? errors[row] * sigmoid : errors[row];
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
			int siz = n1 - (n1 & 7);
			int mi = dMatrixInfo[i][1];
			for (int j = 0; j < siz; j = j + 8) {
				if (isLast) {
					sum += _sums(&yList[yCache + j], &neurons[nCache1 + j]);
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
				float eVal8 = e[index];
				float *d2 = &(d[mi * m++]);
				float *d22 = &(d[mi * m++]);
				float *d23 = &(d[mi * m++]);
				float *d24 = &(d[mi * m++]);
				float *d25 = &(d[mi * m++]);
				float *d26 = &(d[mi * m++]);
				float *d27 = &(d[mi * m++]);
				float *d28 = &(d[mi * m]);
				int size = n2 - (n2 & 7);
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
				for (int d = size; d < n2; d++) {
					float nVal = n[d];
					d2[d] = fma(eVal, nVal, d2[d]);
					d22[d] = fma(eVal2, nVal, d22[d]);
					d23[d] = fma(eVal3, nVal, d23[d]);
					d24[d] = fma(eVal4, nVal, d24[d]);
					d25[d] = fma(eVal5, nVal, d25[d]);
					d26[d] = fma(eVal6, nVal, d26[d]);
					d27[d] = fma(eVal7, nVal, d27[d]);
					d28[d] = fma(eVal8, nVal, d28[d]);
				}
			}

			for (int a = siz; a < n1; a++) {
				sum += isLast ? -(yList[yCache + a] * log(neurons[nCache1 + a])) - ((1 - yList[yCache + a]) * log(1 - neurons[nCache1 + a])) : 0;
				int index = i == 0 ? a + 1 : a;
				float eVal = e[index];
				float *d2 = &(d[mi * a]);
				int size = n2 - (n2 & 7);
				for (int d = 0; d < size; d += 8) {
					_mulAddBroadcast(&d2[d], &eVal, &n[d]);
				}
				for (int d = size; d < n2; d++) {
					d2[d] = fma(eVal, n[d], d2[d]);
				}
			}

		}
		data->cost = fma(data->ySizeF, sum, data->cost);

	}

	free(neurons);
	free(errors);
}
void NeuralNetwork::handleWork(stData *param) {
	switch (param->workType) {
	case 12:
		calculateCost(param);
		break;
	case 1:
		fWork1(param);
		break;
	case 2:
		fWork2(param);
		break;
	case 3:
		fWork3(param);
		break;
	case 4:
		fWork4(param);
		break;
	case 5:
		fWork5(param);
		break;
	case 6:
		fWork6(param);
		break;
	case 7:
		fWork7(param);
		break;
	case 8:
		fWork8(param);
		break;
	case 9:
		fWork9(param);
		break;
	case 10:
		fWork10(param);
		break;
	case 11:
		fWork11(param);
		break;
	case 13:
		fWork13(param);
		break;
	default:
		break;
	}
}
void* NeuralNetwork::calculateBackCost(void *dat) {
	struct stData *data = (struct stData*) dat;
	int w = 0;

	for (;;) {
		pthread_mutex_lock(&data->mutex);

		while ((w = data->workType) == 0) {
			pthread_cond_wait(&data->waitCond, &data->mutex);
		}

		pthread_mutex_unlock(&data->mutex);

		if (w == -1) {
			break;
		}

		handleWork(data);

		pthread_mutex_lock(&data->mutex);
		data->workType = 0;
		pthread_mutex_unlock(&data->mutex);
		pthread_cond_signal(&data->completeCond);
	}
	pthread_exit(NULL);

	return 0;
}
void NeuralNetwork::submitWork(int workType) {
	for (int t = numberOfThreads - 1; t >= 0; t--) {
		if (stDatas[t].isMain) {
			//if its last handle by main thread
			stDatas[t].workType = workType;
			handleWork(&stDatas[t]);

		} else {
			pthread_mutex_lock(&stDatas[t].mutex);
			stDatas[t].workType = workType;
			pthread_mutex_unlock(&stDatas[t].mutex);
			pthread_cond_signal(&stDatas[t].waitCond);
		}

	}
	for (int t = numberOfThreads - 1; t > 0; t--) {
		pthread_mutex_lock(&stDatas[t].mutex);
		while (stDatas[t].workType != 0) {
			pthread_cond_wait(&stDatas[t].completeCond, &stDatas[t].mutex);
		}
		pthread_mutex_unlock(&stDatas[t].mutex);
	}
}
float NeuralNetwork::calculateBackCostWithThetas(float *thetas) {
//allocate place for deltas
	free(deltas);
	deltas = (float*) malloc(mDeltaSize);
//create params for each thread

	float cost = 0.0f;
	for (int t = numberOfThreads - 1; t >= 0; t--) {
		stDatas[t].thetas = thetas;
		stDatas[t].cost = 0.0f;
		stDatas[t].calculatedDeltas = &(deltas[stDatas[t].tloopmin]);
		if (stDatas[t].isLast) {
			//if its last handle by main thread
			calculateCost(&stDatas[t]);
		} else {
			pthread_mutex_lock(&stDatas[t].mutex);
			stDatas[t].workType = 12;
			pthread_mutex_unlock(&stDatas[t].mutex);
			pthread_cond_signal(&stDatas[t].waitCond);
		}

	}
	for (int t = numberOfThreads - 1; t > 0; t--) {
		pthread_mutex_lock(&stDatas[t].mutex);
		while (stDatas[t].workType != 0) {
			pthread_cond_wait(&stDatas[t].completeCond, &stDatas[t].mutex);
		}
		pthread_mutex_unlock(&stDatas[t].mutex);
	}

	float thetaSum = 0.0;

//collect all data from threads and update cost

	int da = 0;
	int dc = 0;
	for (int l = 0; l < deltaSize; l++) {
		dc = (l - dLayerCache[da]) % dMatrixDimensions[da][1];
		deltas[l] = 0.0;
		for (int i = 0; i < numberOfThreads; i++) {
			deltas[l] += stDatas[i].deltas[l];
		}
		deltas[l] *= yf;
		deltas[l] = dc > 0 ? fma(lyf, thetas[l], deltas[l]) : deltas[l];
		thetaSum += dc > 0 ? pow(thetas[l], 2) : 0;
		da += (l + 1) == dLayerCache[da + 1];
	}

	for (int i = 0; i < numberOfThreads; ++i) {
		cost += stDatas[i].cost;
	}

	return fma(thetaSum, tyf, cost);

}

void NeuralNetwork::predict(float *tList, float *yTemp) {

	int totalCorrect = 0;
	int totalWrong = 0;

	for (int i = 0; i < ySize; ++i) {

		float *neurons = forwardPropogate(tList, &(xList[(i * xColumns)]));
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
float* NeuralNetwork::forwardPropogate(float *tList, float *xList) {

	int mNeuronSize = sizeof(float) * neuronSize;
	float *neurons = (float*) malloc(mNeuronSize);
	for (int l = 0; l < layerCount; l++) {
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
				neurons[row] = xList[jPrev];
			} else {
				int lPrev = l - 1;
				int dCache = dLayerCache[l - 1];
				int nCounts = neuronCounts[lPrev] + 1;
				int siz = nCounts - (nCounts & 7);
				float *n = &(neurons[nLayerCache[lPrev]]);
				float *t = &(tList[(dMatrixDimensions[lPrev][1] * (isLast ? j : jPrev)) + dCache]);
				for (int k = 0; k < siz; k = k + 8) {
					neurons[row] += _mulAdd(&t[k], &n[k]);
				}
				for (int k = siz; k < nCounts; k++) {
					neurons[row] = fma(t[k], n[k], neurons[row]);
				}

				neurons[row] = (UPPER_BOUND / (1 + pow(E, -neurons[row]))) + LOWER_BOUND;
			}
		}
	}

	return neurons;
}

