/*
 * ApplicationParameters.h
 *
 *  Created on: Feb 13, 2015
 *      Author: ubuntu
 */

#ifndef APPLICATIONPARAMETERS_H_
#define APPLICATIONPARAMETERS_H_
#include <string>
#include <stdio.h>
#include <stdlib.h>

using namespace std;
typedef  long long int llu;
class ApplicationParameters {

private:
	string xPath;
	string yPath;
	string tPath;
	llu rowCount;
	llu colCount;
	llu numberOfLabels;
	llu totalLayerCount;
	llu *hiddenLayerSize;
	llu numberOfThreads;
	llu maxIteration;
	float lambda;
	llu predict;
	llu loadThetas;
	llu saveThetas;
	llu scale;
	llu validCount;
	llu valid;
	llu testPercentage;
	llu predictionStep;
	llu cpus;
	llu random;

	void validateInputs(llu argc, char **argv);
public:
	ApplicationParameters(llu argc, char **argv);
	virtual ~ApplicationParameters();
	/*
	 * Get path of input list
	 */
	string getXPath();
	/*
	 * Get path of expectation list
	 */
	string getYPath();
	/*
	 * get path of previously saved thetas
	 */
	string getThetasPat();
	/*
	 * Get row count of X or Y list
	 */
	llu getRowCount();
	/*
	 * Get column count of X list
	 */
	llu getColumnCount();
	/*
	 * Get number of labels (Quantitiy of expectations)
	 */
	llu getNumberOfLabels();
	/*
	 * Get total layer count
	 */
	llu getTotalLayerCount();
	/*
	 * Get hidden layer size
	 */
	llu *getHiddenLayerSize();
	/*
	 * Get number of threads
	 */
	llu getNumberOfThreads();
	/*
	 * Get maximum iterations
	 */
	llu getMaxIteration();
	/*
	 * Get momentum
	 */
	llu getLambda();
	/*
	 * Get if user ask application to\n
	 * do prediction after training complete
	 */
	llu isCrossPredictionEnabled();
	llu loadThetasEnabled();
	llu saveThetasEnabled();
	llu scaleInputsEnabled();
	void printHelp();
	llu isValid();

	llu getTestPercentage();

	void setRowCount(llu count);

	llu steps();

	llu getCpus();
	llu isRandom();

};

#endif /* APPLICATIONPARAMETERS_H_ */
