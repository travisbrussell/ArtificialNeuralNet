#pragma once
#include "ANNLayer.h"
#include "Config.h"
#include "DataPoint.h"
#include <math.h>

class ANN
{
public:
	ANN();
	~ANN();
	void TrainANN(ANNLayer** net, Config* config, DataPoint** trainingPoints);
	void BackProp(Config* config, ANNLayer** net, DataPoint* point, double** outputs);
	double** ReturnOutputs(DataPoint* point, Config* config, ANNLayer** net);
	double Sigmoid(double input);
	double ReturnPrediction(DataPoint* point, Config* config, ANNLayer** net);
};

