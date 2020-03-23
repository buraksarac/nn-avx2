/*
 % Copyright (C) 2001 and 2002 by Carl Edward Rasmussen. Date 2002-02-13
 %
 %
 % (C) Copyright 1999, 2000 & 2001, Carl Edward Rasmussen
 %
 % Permission is granted for anyone to copy, use, or modify these
 % programs and accompanying documents for purposes of research or
 % education, provided this copyright notice is retained, and note is
 % made of any changes that have been made.
 %
 % These programs and documents are distributed without any warranty,
 % express or implied.  As the programs were written for research
 % purposes only, they have not been tested to the degree that would be
 % advisable in any important application.  All use of these programs is
 % entirely at the user's own risk.

 Changes made:
 burak sarac : c/c++ implementation
 */

#include "Fmincg.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "GradientParameter.h"
#include <limits>
#include <sys/time.h>
#include <ctime>
#include <deque>
#include <immintrin.h>

using namespace std;
typedef std::numeric_limits<float> lim_dbl;
static const float ALL_SET = -1.0f - lim_dbl::epsilon() / lim_dbl::radix;
static const __m256 V_ALL_SET = _mm256_set1_ps(ALL_SET);
static const float RHO = 0.01; // a bunch of constants for line searches
static const float SIG = 0.5; // RHO and SIG are the constants in the Wolfe-Powell conditions
static const float INT = 0.1; // don't reevaluate within 0.1 of the limit of the current bracket
static const float EXT = 3.0; // extrapolate maximum 3 times the current bracket
static const float MAX = 20.0; // max 20 function evaluations per line search
static const float RATIO = 100; // maximum allowed slope ratio

static NeuralNetwork *neuralNetwork;
NeuralNetwork* Fmincg::getNN() {
	return neuralNetwork;
}
void _mulFmaddStore(float *a, float *b, float *c) {
	__m256 va = _mm256_loadu_ps(a);
	__m256 vb = _mm256_loadu_ps(b);
	__m256 vc = _mm256_loadu_ps(c);
	__m256 vres = _mm256_fmadd_ps(va, vb, vc);
	_mm256_storeu_ps(c, vres);

}

void _mulFmaddStore1(float *a, float *d) {
	__m256 va = _mm256_loadu_ps(a);
	__m256 vres = _mm256_mul_ps(va, va);
	vres = _mm256_mul_ps(V_ALL_SET, vres);
	for (int i = 0; i < 8; i++) {
		d[0] += vres[i];
	}
}
void _mulFmaddStore2(float *a, float *b, float *c) {
	__m256 va = _mm256_loadu_ps(a);
	__m256 vb = _mm256_loadu_ps(b);
	__m256 vres = _mm256_mul_ps(va, vb);
	for (int i = 0; i < 8; i++) {
		c[0] += vres[i];
	}
}
void _mulNegate(float *a, float *d) {
	__m256 va = _mm256_loadu_ps(a);
	__m256 vres = _mm256_mul_ps(va, V_ALL_SET);
	_mm256_storeu_ps(d, vres);
}
void _mcopy(float *a, float *b) {
	__m256 buffer = _mm256_loadu_ps(a);
	_mm256_storeu_ps(b, buffer);
}
void _mulAddBroadcastF(float *d, float *e, float *n) {
	__m256 vd = _mm256_loadu_ps(d);
	__m256 vn = _mm256_loadu_ps(n);
	__m256 ve = _mm256_broadcast_ss(e);

	__m256 res = _mm256_fmadd_ps(ve, vn, vd);
	_mm256_storeu_ps(d, res);
}
void _mulSub(float *a, float *b, float *c) {
	__m256 va = _mm256_broadcast_ss(a);
	__m256 vb = _mm256_loadu_ps(b);
	__m256 vc = _mm256_loadu_ps(c);
	__m256 vres = _mm256_fmsub_ps(va, vb, vc);
	_mm256_storeu_ps(b, vres);
}
void _mswap(float *a, float *b) {
	__m256 va = _mm256_loadu_ps(a);
	__m256 vb = _mm256_loadu_ps(b);
	_mm256_storeu_ps(a, vb);
	_mm256_storeu_ps(b, va);
}
GradientParameter* Fmincg::calculate(int thetaRowCount, int noThreads, int numberOfLabels, int maxIterations, float *aList, int ySize, int xColumnSize, float *yList, int totalLayerCount,
		int *neuronCounts, float lambda, float *yTemp, int testRows, int steps, int save) {

	srand(time(0));

	float nLimit = numeric_limits<float>::epsilon();
	float *thetas = (float*) malloc(sizeof(float) * thetaRowCount);
	int columns = 0;
	for (int i = 0; i < totalLayerCount - 1; i++) {
		for (int j = 0; j < neuronCounts[i + 1]; j++) {
			for (int k = 0; k < neuronCounts[i] + 1; k++) {
				int r = (rand() % neuronCounts[i + 1]) + neuronCounts[i] + 1;
				thetas[columns++] = 0;//r * 2 * nLimit - nLimit;
			}
		}
	}

	return Fmincg::calculate(noThreads, thetaRowCount, numberOfLabels, maxIterations, aList, ySize, xColumnSize, yList, totalLayerCount, neuronCounts, lambda, thetas, yTemp, testRows, steps, save);

}

GradientParameter* Fmincg::calculate(int noThreads, int thetaRowCount, int numberOfLabels, int maxIterations, float *aList, int ySize, int xColumnSize, float *yList, int layerCount, int *neuronCounts,
		float lambda, float *tList, float *yTemp, int testRows, int steps, int save) {

	float *x = tList;

	neuralNetwork = new NeuralNetwork(noThreads, aList, yList, layerCount, neuronCounts, numberOfLabels, ySize, xColumnSize, lambda);
	int i = 0;
	int ls_failed = 0;   // no previous line search has failed
	int n = 0;
	int diff = thetaRowCount & 7;
	int size = thetaRowCount - diff;
//gd instance will change during the iteration
	GradientParameter *gd = neuralNetwork->calculateBackCostWithThetas(x);
	n++;
	float d1 = 0.0; //search direction is steepest and calculate slope
	float f1 = gd->getCost();
	float *df1 = (float*) malloc(sizeof(float) * thetaRowCount);
	float *s = (float*) malloc(sizeof(float) * thetaRowCount);
	deque<float> results;

	for (int r = 0; r < size; r += 8) {
		_mcopy(&(gd->getThetas()[r]), &(df1[r]));
		_mulNegate(&df1[r], &s[r]);
		_mulFmaddStore1(&s[r], &d1);
	}
	for (int r = size; r < thetaRowCount; r++) {
		df1[r] = gd->getThetas()[r];
		s[r] = -1.0f * df1[r];
		d1 += -1.0f * s[r] * s[r];
	}
	delete gd;
	float z1 = 1.0f / (1.0f - d1);

	float *x0 = new float[thetaRowCount];
	float *df0 = new float[thetaRowCount];
	float *df2 = new float[thetaRowCount];
	float d2 = 0.0;
	float f3 = 0.0;

	float A = 0.0;
	float B = 0.0;
	int iter = abs(maxIterations);
	while (i < iter) {
		i++;
		//lets start

		//X0 = X; f0 = f1; df0 = df1; make a copy of current values
		float f0 = f1;

		for (int r = 0; r < size; r += 8) {
			_mcopy(&x[r], &x0[r]);
			_mcopy(&df1[r], &df0[r]);
			_mulAddBroadcastF(&x[r], &z1, &s[r]);
		}

		for (int r = size; r < thetaRowCount; r++) {
			x0[r] = x[r]; //copy x value into x0
			df0[r] = df1[r]; //copy df1 value into df0
			x[r] += z1 * s[r]; //update x as X = X + z1*s;
		}

		//request new gradient after we update X -- octave -->[f2 df2] = eval(argstr);
		GradientParameter *gd2 = neuralNetwork->calculateBackCostWithThetas(x);
		n++;
		float f2 = gd2->getCost();

		d2 = 0.0;
		for (int r = 0; r < size; r += 8) {
			_mcopy(&(gd2->getThetas()[r]), &df2[r]);
			_mulFmaddStore2(&s[r], &df2[r], &d2);
		}
		for (int r = size; r < thetaRowCount; r++) {
			df2[r] = gd2->getThetas()[r];
			d2 += s[r] * df2[r]; // d2 = df2'*s;
		}

		//f3 = f1; d3 = d1; z3 = -z1;        initialize point 3 equal to point 1
		f3 = f1;
		float d3 = d1;
		float z3 = -1 * z1;
		float M = MAX;
		int success = 0;
		float limit = -1;
		delete gd2;
		while (1) {
			float z2 = 0.0;
			while (((f2 > f1 + (z1 * RHO * d1)) | (d2 > (-1 * SIG * d1))) & (M > 0)) {
				limit = z1;

				if (f2 > f1) {
					z2 = z3 - (0.5 * d3 * z3 * z3) / (d3 * z3 + f2 - f3);
				} else {
					A = 6.0f * (f2 - f3) / z3 + 3 * (d2 + d3);
					B = 3.0f * (f3 - f2) - z3 * (d3 + 2 * d2);
					z2 = (sqrt(B * B - A * d2 * z3 * z3) - B) / A;
				}

				if (isnan(z2) | isinf(z2)) {
					z2 = z3 / 2;
				}

				float z3m1 = INT * z3;
				float z3m2 = (1 - INT) * z3;
				float min = z2 < z3m1 ? z2 : z3m1;
				z2 = min > z3m2 ? min : z3m2;
				z1 = z1 + z2;

				for (int r = 0; r < size; r += 8) {
					_mulAddBroadcastF(&x[r], &z2, &s[r]);
				}
				for (int r = size; r < thetaRowCount; r++) {
					x[r] += z2 * s[r];
				}

				GradientParameter *gd3 = neuralNetwork->calculateBackCostWithThetas(x);
				n++;
				M = M - 1;
				f2 = gd3->getCost();

				d2 = 0.0;
				for (int r = 0; r < size; r += 8) {
					_mcopy(&(gd3->getThetas()[r]), &df2[r]);
					_mulFmaddStore2(&s[r], &df2[r], &d2);
				}
				for (int r = size; r < thetaRowCount; r++) {
					df2[r] = gd3->getThetas()[r];
					d2 += s[r] * df2[r]; // d2 = df2'*s;
				}
				delete gd3;
				z3 = z3 - z2;        // z3 is now relative to the location of z2
			}

			if ((f2 > f1 + (z1 * RHO * d1)) | (d2 > -1 * SIG * d1)) {
				break;
			} else if (d2 > SIG * d1) {
				success = 1;
				break;
			} else if (M == 0) {
				break;
			}

			A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3);  // make cubic extrapolation
			B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2);
			z2 = -d2 * z3 * z3 / (B + sqrt(B * B - A * d2 * z3 * z3)); // num. error possible - ok!
			if (isnan(z2) | isinf(z2) | (z2 < 0)) {
				if (limit < -1 * 0.5) {
					z2 = z1 * (EXT - 1.0);
				} else {
					z2 = (limit - z1) / 2.0;
				}
			} else if ((limit > -0.5) & (z2 + z1 > limit)) {
				z2 = (limit - z1) / 2.0;
			} else if ((limit < -0.5) & (z2 + z1 > z1 * EXT)) {
				z2 = z1 * (EXT - 1.0);
			} else if (z2 < -z3 * INT) {
				z2 = -z3 * INT;
			} else if ((limit > -0.5) & (z2 < (limit - z1) * (1.0 - INT))) {
				z2 = (limit - z1) * (1.0 - INT);
			}

			f3 = f2;
			d3 = d2;
			z3 = -1 * z2;
			z1 = z1 + z2;

			for (int r = 0; r < size; r += 8) {
				_mulAddBroadcastF(&x[r], &z2, &s[r]);
			}
			for (int r = size; r < thetaRowCount; r++) {
				x[r] += z2 * s[r];
			}

			GradientParameter *gd4 = neuralNetwork->calculateBackCostWithThetas(x);
			n++;
			M = M - 1;
			f2 = gd4->getCost();
			d2 = 0.0;
			for (int r = 0; r < size; r += 8) {
				_mcopy(&(gd4->getThetas()[r]), &df2[r]);
				_mulFmaddStore2(&s[r], &df2[r], &d2);
			}
			for (int r = size; r < thetaRowCount; r++) {
				df2[r] = gd4->getThetas()[r];
				d2 += s[r] * df2[r]; // d2 = df2'*s;
			}
			delete gd4;
		}

		if (success) {
			f1 = f2;
			results.push_back(f1);

			if (i != 1 && ((i & (steps - 1)) == 0)) {
				printf("\n Next success cost: [[ %0.22f ]] total [[ %i ]] iteration and [[ %i ]] neural calculation complete", f1, i, n);
				float *testXlist = &(aList[(ySize) * xColumnSize]);
				float *testYlist = &(yTemp[ySize]);
				neuralNetwork->predict(testRows, testXlist, x, testYlist);
				if (save) {
					IOUtils::saveThetas(x, thetaRowCount);
				}
			}
			// Polack-Ribiere direction
			float sum1 = 0.0;
			float sum2 = 0.0;
			float sum3 = 0.0;
			for (int r = 0; r < size; r += 8) {
				_mulFmaddStore2(&df2[r], &df2[r], &sum1);
				_mulFmaddStore2(&df1[r], &df2[r], &sum2);
				_mulFmaddStore2(&df1[r], &df1[r], &sum3);
			}
			for (int r = size; r < thetaRowCount; r++) {
				sum1 += df2[r] * df2[r];
				sum2 += df1[r] * df2[r];
				sum3 += df1[r] * df1[r];
			}

			float p = (sum1 - sum2) / sum3;
			d2 = 0.0;
			for (int r = 0; r < size; r += 8) {
				_mulSub(&p, &s[r], &df2[r]);
				_mswap(&df1[r], &df2[r]);
				_mulFmaddStore2(&s[r], &df1[r], &d2);
			}
			for (int r = size; r < thetaRowCount; r++) {
				s[r] = p * s[r] - df2[r];
				float tmp = df1[r];
				df1[r] = df2[r];
				df2[r] = tmp;
				d2 += df1[r] * s[r]; // d2 = df1'*s;
			}

			if (d2 > 0) {
				d2 = 0.0;
				for (int r = 0; r < size; r += 8) {
					_mulNegate(&df1[r], &s[r]);
					_mulFmaddStore1(&s[r], &d2);
				}
				for (int r = size; r < thetaRowCount; r++) {
					s[r] = -1.0f * df1[r]; // s = -df1;
					d2 += -1.0f * s[r] * s[r]; // d2 = -s'*s;
				}
			}

			float sum4 = d1 / (d2 - numeric_limits<float>::min());
			z1 = z1 * (RATIO < sum4 ? RATIO : sum4);
			d1 = d2;
			ls_failed = 0;

		} else {

			f1 = f0;
			for (int r = 0; r < size; r += 8) {
				_mcopy(&x0[r], &x[r]);
				_mcopy(&df0[r], &df1[r]);
			}
			for (int r = size; r < thetaRowCount; r++) {
				x[r] = x0[r];
				df1[r] = df0[r];
			}

			if (ls_failed) {
				break;
			}

			d1 = 0.0;
			for (int r = 0; r < size; r += 8) {
				_mswap(&df1[r], &df2[r]);
				_mulNegate(&df1[r], &s[r]);
				_mulFmaddStore1(&s[r], &d1);
			}
			for (int r = 0; r < thetaRowCount; r++) {
				float tmp = df1[r];
				df1[r] = df2[r];
				df2[r] = tmp;
				s[r] = -1.0f * df1[r];
				d1 += -1.0f * s[r] * s[r];
			}

			z1 = 1.0f / (1.0f - d1);
			ls_failed = 1;

		}

	}

	free(df1);
	free(s);
	delete[] x0;
	delete[] df0;
	delete[] df2;

	//cleaning deltas is GradientParameter job now
	return new GradientParameter(x, results);

}

