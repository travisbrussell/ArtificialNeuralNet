#pragma once

struct Config
{
	Config()
	{
		// constructor
	}

	~Config()
	{
		// destructor
	}

	// parameters for training //
	double LearningRate;
	int NumberOfLayers; // #hidden layers + output layer
	int* LayerSizes; // sizes of hidden layers and output layer
	int NumberOfEpics;

	// data parameters
	int NumberOfAttributes;
	double* MaxContinuousValues; // Max value to consider for random question generation
	double* MinContinuousValues; // Min value to consider for random question generation
	int NumberOfTrainingPoints;
	int NumberOfTestingPoints;
};
