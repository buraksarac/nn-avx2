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

 Changes mage:
 burak sarac : c/c++ implementation
 */

#ifndef SRC_FMINCG_H_
#define SRC_FMINCG_H_

#include "GradientParameter.h"
#include "ApplicationParameters.h"
#include "NeuralNetwork.h"
#include "IOUtils.h"
typedef  long long int llu;
class Fmincg {
public:
	static NeuralNetwork* getNN();
	static GradientParameter* calculate(ApplicationParameters *params, llu thetaRowCount, float *aList, float *yList, llu *neuronCounts, float *yTemp, llu testRows);
	static GradientParameter* calculate(ApplicationParameters *params, llu thetaRowCount, float *aList, float *yList, llu *neuronCounts, float *tList, float *yTemp, llu testRows);
};

#endif /* SRC_FMINCG_H_ */

