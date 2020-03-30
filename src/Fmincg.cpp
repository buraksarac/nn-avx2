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
#include <thread>

using namespace std;
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
				thetas[columns++] = r * 2 * nLimit - nLimit;
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
//gd instance will change during the iteration
	n++;

	float *df1 = (float*) malloc(sizeof(float) * thetaRowCount);
	float *s = (float*) malloc(sizeof(float) * thetaRowCount);
	float *x0 = new float[thetaRowCount];
	float *df0 = new float[thetaRowCount];
	float *df2 = new float[thetaRowCount];
	struct stData *fParam = neuralNetwork->stDatas;
	unsigned concurentThreadsSupported = std::thread::hardware_concurrency();
	int threadBarrier = concurentThreadsSupported - noThreads;
	for (int a = concurentThreadsSupported - 1; a >= threadBarrier; a--) {
		int t = a - threadBarrier;
		int loopmin = (int) ((long) (t + 0) * (long) (thetaRowCount) / (long) noThreads);
		int loopmax = (int) ((long) (t + 1) * (long) (thetaRowCount) / (long) noThreads);
		int length = loopmax - loopmin;
		fParam[t].x = &(x[loopmin]);
		fParam[t].x0 = &(x0[loopmin]);
		fParam[t].df1 = &(df1[loopmin]);
		fParam[t].df0 = &(df0[loopmin]);
		fParam[t].df2 = &(df2[loopmin]);
		fParam[t].s = &(s[loopmin]);
		fParam[t].calculatedDeltas = &(neuralNetwork->deltas[loopmin]);
		fParam[t].end = length;
		fParam[t].size = length - (length & 7);
		fParam[t].isMain = t == 0;
		fParam[t].z1 = 0;
		fParam[t].d1 = 0;
		fParam[t].d2 = 0;
		fParam[t].z2 = 0;
		fParam[t].sum1 = 0;
		fParam[t].sum2 = 0;
		fParam[t].sum3 = 0;
		fParam[t].p = 0;
		fParam[t].tloopmin = loopmin;

	}
	float f1 = neuralNetwork->calculateBackCostWithThetas(x);
	deque<float> results;
	float d1 = 0.0;
	neuralNetwork->submitWork(13);
	for (int i = noThreads - 1; i >= 0; i--) {
		d1 += fParam[i].d1;
	}
	float z1 = 1.0f / (1.0f - d1);

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

		//work1
		for (int i = noThreads - 1; i >= 0; i--) {
			fParam[i].z1 = z1;
		}
		neuralNetwork->submitWork(1);

		//request new gradient after we update X -- octave -->[f2 df2] = eval(argstr);
		n++;
		float f2 = neuralNetwork->calculateBackCostWithThetas(x);

		//work2
		d2 = 0.0;
		neuralNetwork->submitWork(2);
		for (int i = noThreads - 1; i >= 0; i--) {
			d2 += fParam[i].d2;
		}

		//f3 = f1; d3 = d1; z3 = -z1;        initialize point 3 equal to point 1
		f3 = f1;
		float d3 = d1;
		float z3 = -1 * z1;
		float M = MAX;
		int success = 0;
		float limit = -1;
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

				//work3
				for (int i = noThreads - 1; i >= 0; i--) {
					fParam[i].z2 = z2;
				}
				neuralNetwork->submitWork(3);

				n++;
				M = M - 1;
				f2 = neuralNetwork->calculateBackCostWithThetas(x);

				//work4
				d2 = 0.0;
				neuralNetwork->submitWork(4);
				for (int i = noThreads - 1; i >= 0; i--) {
					d2 += fParam[i].d2;
				}
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

			//work5
			for (int i = noThreads - 1; i >= 0; i--) {
				fParam[i].z2 = z2;
			}
			neuralNetwork->submitWork(5);

			n++;
			M = M - 1;
			f2 = neuralNetwork->calculateBackCostWithThetas(x);
			//work6
			d2 = 0.0;
			neuralNetwork->submitWork(6);
			for (int i = noThreads - 1; i >= 0; i--) {
				d2 += fParam[i].d2;
			}
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

			//work7
			float sum1 = 0.0;
			float sum2 = 0.0;
			float sum3 = 0.0;
			neuralNetwork->submitWork(7);
			for (int i = noThreads - 1; i >= 0; i--) {
				sum1 += fParam[i].sum1;
				sum2 += fParam[i].sum2;
				sum3 += fParam[i].sum3;
			}

			//work8
			float p = (sum1 - sum2) / sum3;
			for (int i = noThreads - 1; i >= 0; i--) {
				fParam[i].p = p;
			}

			d2 = 0.0;
			neuralNetwork->submitWork(8);
			for (int i = noThreads - 1; i >= 0; i--) {
				d2 += fParam[i].d2;
			}

			//work9
			if (d2 > 0) {
				d2 = 0.0;
				neuralNetwork->submitWork(9);
				for (int i = noThreads - 1; i >= 0; i--) {
					d2 += fParam[i].d2;
				}
			}

			float sum4 = d1 / (d2 - numeric_limits<float>::min());
			z1 = z1 * (RATIO < sum4 ? RATIO : sum4);
			d1 = d2;
			ls_failed = 0;

		} else {

			//work10
			f1 = f0;
			neuralNetwork->submitWork(10);

			if (ls_failed) {
				break;
			}

			//work11
			d1 = 0.0;
			neuralNetwork->submitWork(11);
			for (int i = noThreads - 1; i >= 0; i--) {
				d1 += fParam[i].d1;
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

