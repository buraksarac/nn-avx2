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

#include "IOUtils.h"
#include "Fmincg.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "ApplicationParameters.h"
struct timespec tstart = { 0, 0 }, tend = { 0, 0 };
using namespace std;
int main(int argc, char **argv) {

	//parse and validate input parameters
	ApplicationParameters *params = new ApplicationParameters(argc, argv);

	if (!params->isValid()) {
		return 0;
	}

	setbuf(stdout, NULL);
	printf("Start!\n");
	float *x;
	float *xlist;
	float *yTemp;

	try {
		//get file content as array
		x = IOUtils::getArray(params->getXPath(), params->getRowCount(), params->getColumnCount());
		//if user set scale option create featured list
		xlist = params->scaleInputsEnabled() ? IOUtils::getFeaturedList(x, params->getColumnCount(), params->getRowCount()) : x;
		//get expectation list
		yTemp = IOUtils::getArray(params->getYPath(), params->getRowCount(), 1);
	} catch (int e) {
		string message;
		switch (e) {
		case 1:
			message = "Input file doesnt have specified rowcount";
			break;
		case 2:
			message = "Input file content can not be parsed as float or float";
			break;
		case 3:
			message = "System couldnt read input file";
			break;
		default:
			break;
		}
		return 0;
	} catch (...) {
		printf("System couldnt create float array from -x input file. Does file content correct?");
		return 0;
	}

	float *ylist = new float[params->getRowCount() * params->getNumberOfLabels()];

	//parse expected list to 1 and 0.
	//i.e. if value is 3 and number of labels 6
	//it should look like: 0 0 1 0 0 0
	for (int r = 0; r < params->getRowCount(); r++) {
		for (int c = 0; c < params->getNumberOfLabels(); c++) {
			ylist[(r * params->getNumberOfLabels()) + c] = ((c + 1) == abs(yTemp[r])) ? 1 : 0;
		}
	}

	int testRows = 0;
	if (params->getTestPercentage() != 0) {
		testRows = (params->getTestPercentage() * params->getRowCount()) / 100;
		params->setRowCount(params->getRowCount() - testRows);
		printf("\n\n Total %i rows will be trained \n", params->getRowCount());
		printf("\n\n Total %i rows will be tested \n", testRows);
	}

	//collect layer item infos in an array
	int *neuronCount = (int*) malloc(sizeof(int) * params->getTotalLayerCount());
	neuronCount[0] = params->getColumnCount();
	for (int j = 1; j < params->getTotalLayerCount() - 1; ++j) {
		neuronCount[j] = params->getHiddenLayerSize()[j - 1];
	}
	neuronCount[params->getTotalLayerCount() - 1] = params->getNumberOfLabels();

	//calculate weights size
	float *tList;
	float thetaRowCount = 0;
	for (int i = 0; i < params->getTotalLayerCount() - 1; i++) {
		for (int j = 0; j < neuronCount[i + 1]; j++) {
			for (int k = 0; k < neuronCount[i] + 1; k++) {
				thetaRowCount++;
			}
		}
	}
	GradientParameter *gd;
	clock_gettime(CLOCK_MONOTONIC, &tstart);
	//check if user will continue from previously saved training data
	if (params->loadThetasEnabled()) {
		//load thetas
		tList = IOUtils::getArray(params->getThetasPat(), thetaRowCount, 1);

		//start iteration
		gd = Fmincg::calculate(params->getNumberOfThreads(), thetaRowCount, params->getNumberOfLabels(), params->getMaxIteration(), xlist, params->getRowCount(), params->getColumnCount(), ylist,
				params->getTotalLayerCount(), neuronCount, params->getLambda(), tList, yTemp, testRows, params->steps(), params->saveThetasEnabled());

	} else {

		//start iteration
		gd = Fmincg::calculate(thetaRowCount, params->getNumberOfThreads(), params->getNumberOfLabels(), params->getMaxIteration(), xlist, params->getRowCount(), params->getColumnCount(), ylist,
				params->getTotalLayerCount(), neuronCount, params->getLambda(), yTemp, testRows, params->steps(), params->saveThetasEnabled());
	}
	clock_gettime(CLOCK_MONOTONIC, &tend);

	printf("\n\n\t\t >>>>Process took: %.5f second<<<< \n", ((float) tend.tv_sec + 1.0e-9 * tend.tv_nsec) - ((float) tstart.tv_sec + 1.0e-9 * tstart.tv_nsec));

	NeuralNetwork *neuralNetwork = Fmincg::getNN();
	if (params->isCrossPredictionEnabled() && gd->getCosts().size() > 0) {
		printf("\n Final Prediction will start. Calculated cost: %0.50f", gd->getCosts().back());
		if (params->getTestPercentage() == 0) {
			neuralNetwork->predict(gd->getThetas(), yTemp);
		} else {
			float *testXlist = &(xlist[(params->getRowCount()) * params->getColumnCount()]);
			float *testYlist = &(yTemp[params->getRowCount()]);
			neuralNetwork->predict(testRows, testXlist, gd->getThetas(), testYlist);
		}
	}
	//Save thetas if requested
	if (params->saveThetasEnabled()) {
		IOUtils::saveThetas(gd->getThetas(), thetaRowCount);
	}
	neuralNetwork->~NeuralNetwork();
	free(yTemp);
	free(neuronCount);
	delete gd;
	delete params;
	printf("\nFinish!\n");
}

