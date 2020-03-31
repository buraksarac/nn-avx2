/*
 * ApplicationParameters.cpp
 *
 *  Created on: Feb 13, 2015
 *      Author: ubuntu
 */

#include "ApplicationParameters.h"
#include "IOUtils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <thread>
/*
 * Pojo class to hold application parameters and validate
 */
ApplicationParameters::ApplicationParameters(int argc, char **argv) {

	validateInputs(argc, argv);

}

ApplicationParameters::~ApplicationParameters() {
}

void ApplicationParameters::printHelp() {
	printf("\nUSAGE:\n");
	printf("\n--help\tThis help info\n");
	printf("\n-x\tX(input) file path\n");
	printf("\n-y\tY(output, expected result) file path\n");
	printf("\n-r\tRowcount of X or Y file (should be equal)\n");
	printf("\n-c\tColumn count of X file (each row should have same count)\n");
	printf("\n-n\tNumber of labels in Y file (how many expected result, should be a sequence starting from 1)\n");
	printf("\n-t\tTotal hidden layer count (excluding input and output)\n");
	printf("\n-h\tHidden layer size (excluding bias unit)\n");
	printf("\n-j\tNumber of cores(threads) on host pc\n");
	printf("\n-i\tNumber of iteration for training\n");
	printf("\n-l\tLambda value\n");
	printf(
			"\n-cpus\tTotal cpu count on system, if system not able to report total numbers due to isolation set this number to actual total, still -j parameter will be considered but this param will make affinity reliable\n");
	printf("\n-f\tScale inputs for featured list, 0 or 1, optional, default 0)\n");
	printf("\n-p\tDo prediction for each input after training complete (0 for disable 1 for enable default 1)\n");
	printf("\n-tp\tTheta path. If you have previously saved a prediction result you can continue"
			"\n\tfrom this result by loading from file path. (-lt value should be 1)\n");
	printf("\n-lt\tLoad previously saved thetas (prediction result)"
			"\n\t(0 for disable 1 for enable default 0) (-tp needs to be set)\n");
	printf("\n-st\tSave thetas (prediction result)(0 for disable 1 for enable default 1)\n");
	printf("\n-test\tTest percentage, i.e. for 1000 row of data, 10 will result, 900 of row for training and 100 for test\n");
	printf(
			"\n-ps\tPrediction step, has to be power of 2, for long running tasks you can enable this and -st parameter. I.e. -ps 16 will result every 16 iteration will run prediction against test and if -st 1 then also weights will be saved for this prediction, that later you can load back\n");
	printf("\n");
}
void ApplicationParameters::validateInputs(int argc, char **argv) {

	//set default values
	this->numberOfThreads = 1;
	this->maxIteration = 1;
	this->lambda = 1;
	this->predict = 1;
	this->loadThetas = 0;
	this->saveThetas = 0;
	this->scale = 0;
	this->validCount = 0;
	this->valid = 1;
	this->testPercentage = 0;
	this->predictionStep = 32;
	this->cpus = std::thread::hardware_concurrency();

	//Check param size is a odd value
	if ((argc % 1) != 0) {
		printf("Invalid parameter size");
		this->valid = 0;
	}

	//validate and set inputs
	for (int i = 1; i < argc; i = i + 2) {
		if (!strcmp(argv[i], "--help")) {

			printHelp(); //print help

			this->valid = 0;
		} else if (!strcmp(argv[i], "-x")) {

			this->xPath = argv[i + 1]; //input path

			if (!IOUtils::fileExist(this->xPath)) { //check if file exist
				printf("-x parameter %s file doesnt exist!", this->xPath.c_str());
				this->valid = 0;
			}
			this->validCount++;
		} else if (!strcmp(argv[i], "-y")) {

			this->yPath = argv[i + 1]; //expectation list path

			if (!IOUtils::fileExist(this->yPath)) { // check if file exist

				printf("-y parameter %s file doesnt exist!", this->yPath.c_str());
				this->valid = 0;
			}
			this->validCount++;

		} else if (!strcmp(argv[i], "-test")) {

			testPercentage = atoi(argv[i + 1]);

			if (testPercentage > 100 || testPercentage < 0) {

				printf("-tp : testpercentage is not in range of 0-100!");
				this->valid = 0;
			}
			this->validCount++;

		} else if (!strcmp(argv[i], "-r")) {

			this->rowCount = atoi(argv[i + 1]); //row count of x or y list

			if (this->rowCount < 10) { // minimum 10 row

				printf("Rowcount two small");
				this->valid = 0;
			}
			this->validCount++;
		} else if (!strcmp(argv[i], "-cpus")) {

			this->cpus = atoi(argv[i + 1]); //row count of x or y list

		} else if (!strcmp(argv[i], "-ps")) {

			this->predictionStep = atoi(argv[i + 1]); //row count of x or y list
			if ((this->predictionStep & (this->predictionStep - 1)) != 0) {
				printf("Prediction steps can only be power of two (i.e. one of: 2 4 8 16 ...)");
				this->valid = 0;
			}
			this->validCount++;
		} else if (!strcmp(argv[i], "-c")) {
			this->colCount = atoi(argv[i + 1]);
			if (this->colCount < 1) {
				printf("Column count (-c) two small");
				this->valid = 0;
			}
			this->validCount++;
		} else if (!strcmp(argv[i], "-n")) {
			this->numberOfLabels = atoi(argv[i + 1]);
			if (this->numberOfLabels < 2) {
				if (this->rowCount < 1) {
					printf("Number of labels too small");
					this->valid = 0;
				}
			}
			this->validCount++;
		} else if (!strcmp(argv[i], "-t")) {

			this->totalLayerCount = atoi(argv[i + 1]) + 2;

			if (this->totalLayerCount < 3) {

				printf("Total layer count should be greater than 2");
				this->valid = 0;
			}
			this->validCount++;
		} else if (!strcmp(argv[i], "-h")) {
			this->hiddenLayerSize = IOUtils::parseHiddenLayers(argv[i + 1], totalLayerCount);
			this->validCount++;
		} else if (!strcmp(argv[i], "-j")) {
			this->numberOfThreads = atoi(argv[i + 1]);
			if (this->numberOfThreads < 1) {
				printf("Wrong thread set");
				this->valid = 0;
			}
		} else if (!strcmp(argv[i], "-i")) {
			this->maxIteration = atoi(argv[i + 1]);
			if (this->maxIteration < 1) {
				printf("Wrong maxIteration set");
				this->valid = 0;
			}
		} else if (!strcmp(argv[i], "-f")) {
			this->scale = atoi(argv[i + 1]);
			if (!(this->scale == 0 || this->scale == 1)) {
				printf("Scale should be 1 or 0");
				this->valid = 0;
			}
		} else if (!strcmp(argv[i], "-l")) {
			this->lambda = atof(argv[i + 1]);
			if (!(this->lambda >= 0 && this->lambda <= 1)) {
				printf("Lambda should be between 1 and 0");
				this->valid = 0;
			}
		} else if (!strcmp(argv[i], "-p")) {
			this->predict = atoi(argv[i + 1]);
		} else if (!strcmp(argv[i], "-tp")) {
			this->tPath = argv[i + 1];
			if (!IOUtils::fileExist(this->tPath)) {
				printf("-t parameter %s file doesnt exist!", tPath.c_str());
				this->valid = 0;
			}
		} else if (!strcmp(argv[i], "-lt")) {
			this->loadThetas = atoi(argv[i + 1]);
			if (!(this->loadThetas == 0 || this->loadThetas == 1)) {
				printf("loadThetas should be 1 or 0");
				this->valid = 0;
			}
		} else if (!strcmp(argv[i], "-st")) {
			this->saveThetas = atoi(argv[i + 1]);
			if (!(this->saveThetas == 0 || this->saveThetas == 1)) {
				printf("saveThetas should be 1 or 0");
				this->valid = 0;
			}
		} else {
			printf("Couldnt recognize user input");
			this->valid = 0;
		}

	}

	//make sure all 7 required params set
	if (this->validCount < 8) {
		printf("Bad parameters. You need to set all required parameters!\n");
		this->valid = 0;
	}

}

string ApplicationParameters::getXPath() {
	return this->xPath;
}

string ApplicationParameters::getYPath() {
	return this->yPath;
}

string ApplicationParameters::getThetasPat() {
	return this->tPath;
}

int ApplicationParameters::getRowCount() {
	return this->rowCount;
}

int ApplicationParameters::getColumnCount() {
	return this->colCount;
}

int ApplicationParameters::getNumberOfLabels() {
	return this->numberOfLabels;
}

int ApplicationParameters::getNumberOfThreads() {
	return this->numberOfThreads;
}

int ApplicationParameters::getTotalLayerCount() {
	return this->totalLayerCount;
}

int* ApplicationParameters::getHiddenLayerSize() {
	return this->hiddenLayerSize;
}

int ApplicationParameters::getMaxIteration() {
	return this->maxIteration;
}

int ApplicationParameters::getLambda() {
	return this->lambda;
}

int ApplicationParameters::isCrossPredictionEnabled() {
	return this->predict;
}

int ApplicationParameters::loadThetasEnabled() {
	return this->loadThetas;
}

int ApplicationParameters::saveThetasEnabled() {
	return this->saveThetas;
}

int ApplicationParameters::scaleInputsEnabled() {
	return this->scale;
}

int ApplicationParameters::isValid() {
	return this->valid;
}

int ApplicationParameters::getTestPercentage() {
	return this->testPercentage;
}

void ApplicationParameters::setRowCount(int count) {
	this->rowCount = count;
}

int ApplicationParameters::steps() {
	return this->predictionStep;
}

int ApplicationParameters::getCpus(){
	return this->cpus;
}

