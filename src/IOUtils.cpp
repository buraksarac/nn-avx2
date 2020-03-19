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
#include <iostream>
#include <cmath>
#include <math.h>
#include <string>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <sys/time.h>
#include <sys/stat.h>
#include <assert.h>
#include <string.h>

IOUtils::IOUtils() {
	// TODO Auto-generated constructor stub

}

int IOUtils::fileExist(string name) {
	struct stat buffer;
	return (stat(name.c_str(), &buffer) == 0);

}
float* IOUtils::getFeaturedList(float *list, int columnSize, int rowSize) {

	float *sums = (float*) malloc(sizeof(float) * rowSize);
	float *means = (float*) malloc(sizeof(float) * rowSize);
	float *stds = (float*) malloc(sizeof(float) * rowSize);
	float *featuredList = (float*) malloc(sizeof(float) * rowSize * columnSize);

	for (int i = 0; i < rowSize; ++i) {
		float sum = 0.0;
		float correction = 0.0;
		for (int j = 0; j < columnSize; ++j) {
			float y = list[(i * columnSize) + j] - correction;
			float t = sum + y;
			correction = (t - sum) - y;
			sum = t;
		}
		sums[i] = sum;
		means[i] = sums[i] / columnSize;
	}

	for (int i = 0; i < rowSize; ++i) {
		float sum = 0.0;
		float correction = 0.0;
		for (int j = 0; j < columnSize; ++j) {
			float value = std::pow((list[(i * columnSize) + j] - means[i]), 2);
			float y = value - correction;
			float t = sum + y;
			correction = (t - sum) - y;
			sum = t;
		}
		stds[i] = sum;
	}

	for (int i = 0; i < rowSize; ++i) {
		stds[i] = sqrt(stds[i] / columnSize);
	}

	for (int i = 0; i < rowSize; ++i) {
		for (int j = 0; j < columnSize; ++j) {
			featuredList[(i * columnSize) + j] = (list[(i * columnSize) + j] - means[i]) / stds[i];
		}
	}

	free(sums);
	free(means);
	free(stds);
	return featuredList;
}
void IOUtils::saveThetas(float *thetas, lint size) {
	struct timespec tstart = { 0, 0 };
	clock_gettime(CLOCK_MONOTONIC, &tstart);
	string fileName = "thetas_";
	std::stringstream sstm;
	sstm << fileName << tstart.tv_sec << tstart.tv_nsec << ".dat";

	ofstream f(sstm.str().c_str());
	copy(thetas, thetas + size, ostream_iterator<float>(f, "\n"));
	printf("\t\t|\n\t\t\\__Thetas (%s) has been saved into project folder.\n", sstm.str().c_str());
}

float* IOUtils::getArray(string path, lint rows, lint columns) {

	ifstream inputStream;

	lint currentRow = 0;
	std::string s;
	inputStream.open(path.c_str());

	if (!inputStream.is_open()) {
		throw 3;
	}
	lint size = columns * rows;
	lint mListSize = sizeof(float) * size;
	float *list = (float*) malloc(mListSize);

	while (!inputStream.eof()) {

		if (currentRow < size) {

			inputStream >> s;
			try {
				list[currentRow++] = strtod(s.c_str(), NULL);
			} catch (...) {
				throw 2;
			}

		} else {
			break;
		}
	}

	inputStream.close();

	if (currentRow < (size - 1)) {
		throw 1;
	}
	return list;
}
char** IOUtils::str_split(char *a_str, const char a_delim) {
	char **result = 0;
	size_t count = 0;
	char *tmp = a_str;
	char *last_comma = 0;
	char delim[2];
	delim[0] = a_delim;
	delim[1] = 0;

	/* Count how many elements will be extracted. */
	while (*tmp) {
		if (a_delim == *tmp) {
			count++;
			last_comma = tmp;
		}
		tmp++;
	}

	/* Add space for trailing token. */
	count += last_comma < (a_str + strlen(a_str) - 1);

	/* Add space for terminating null string so caller
	 knows where the list of returned strings ends. */
	count++;

	result = (char**) malloc(sizeof(char*) * count);

	if (result) {
		size_t idx = 0;
		char *token = strtok(a_str, delim);

		while (token) {
			assert(idx < count);
			*(result + idx++) = strdup(token);
			token = strtok(0, delim);
		}
		assert(idx == count - 1);
		*(result + idx) = 0;
	}

	return result;
}

int* IOUtils::parseHiddenLayers(char *str, int size) {
	char **tokens;
	tokens = str_split(str, ',');

	if (tokens) {
		int *list = (int *) malloc(sizeof(int)*size);
		int i;
		for (i = 0; *(tokens + i); i++) {
			list[i] = atoi(*(tokens + i));
			free(*(tokens + i));
		}
		free(tokens);
		return list;
	}
	return 0;
}
