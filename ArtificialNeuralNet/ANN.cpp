#include "stdafx.h"
#include "ANN.h"


ANN::ANN()
{
}


ANN::~ANN()
{
}

void ANN::TrainANN(ANNLayer** net, Config* config, DataPoint** trainingPoints)
{
	// train the net
	for (int k = 0; k < config->NumberOfEpics; k++)
	{
		for (int i = 0; i < config->NumberOfTrainingPoints; i++)
		{
			double** outputs = ReturnOutputs(trainingPoints[i], config, net);
			BackProp(config, net, trainingPoints[i], outputs);
			delete[] outputs;
		}
	}
}

void ANN::BackProp(Config* config, ANNLayer** net, DataPoint* point, double** outputs)
{
	int numberOfLayers = config->NumberOfLayers;
	double eta = config->LearningRate;
	int* layerSizes = config->LayerSizes;

	double prediction = outputs[numberOfLayers - 1][0];
	double* deltas = new double[1]; // will update this vector at each iteration.
	deltas[0] = prediction * (1 - prediction) * (point->ClassName - prediction);

	// update all but first layer
	for (int k = 1; k < numberOfLayers; k++)
	{
		// initialize next delta vector;
		double* nextDeltas = new double[net[numberOfLayers - k]->NumberOfInputNodes];
		for (int l = 0; l < net[numberOfLayers - k]->NumberOfInputNodes; l++) nextDeltas[l] = 0;

		// update weights and deltas
		for (int i = 0; i < net[numberOfLayers - k]->NumberOfOutputNodes; i++)
		{
			for (int j = 0; j < net[numberOfLayers - k]->NumberOfInputNodes; j++)
			{
				// update weights
				double initialWeight = net[numberOfLayers - k]->weights[i][j];
				double inputToNode = outputs[numberOfLayers - k - 1][j];
				net[numberOfLayers - k]->weights[i][j] = initialWeight + eta * deltas[i] * inputToNode;

				// update next deltas
				nextDeltas[j] += inputToNode * (1 - inputToNode) * initialWeight * deltas[i];
			}
		}

		// update deltas for next iteration
		delete[] deltas;
		deltas = new double[net[numberOfLayers - k]->NumberOfInputNodes];
		for (int l = 0; l < net[numberOfLayers - k]->NumberOfInputNodes; l++) deltas[l] = nextDeltas[l];
		delete[] nextDeltas;
	}

	// update first layer
	for (int i = 0; i < net[0]->NumberOfOutputNodes; i++)
	{
		for (int j = 0; j < net[0]->NumberOfInputNodes; j++)
		{
			// update weights
			double initialWeight = net[0]->weights[i][j];
			double inputToNode = point->Attributes[j];
			net[0]->weights[i][j] = initialWeight + eta * deltas[i] * inputToNode;
		}
	}
}

double** ANN::ReturnOutputs(DataPoint* point, Config* config, ANNLayer** net)
{
	// initialize output values
	double** outputs = new double*[config->NumberOfLayers];
	// set first layer
	/*outputs[0] = new double[config->LayerSizes[0] + 1];
	for (int i = 0; i < net[0]->NumberOfOutputNodes; i++)
	{
		for (int j = 0; j < net[0]->NumberOfInputNodes; j++)
		{
			outputs[0][i] = net[0]->weights[i][j] * point->Attributes[j];
		}
		outputs[0][net[0]->NumberOfOutputNodes] = 1;
	}*/
	for (int k = 0; k < config->NumberOfLayers; k++)
	{
		outputs[k] = new double[net[k]->NumberOfOutputNodes + 1];
		for (int i = 0; i < net[k]->NumberOfOutputNodes; i++)
		{
			outputs[k][i] = 0;
		}
		outputs[k][net[k]->NumberOfOutputNodes] = 1; // threshold placeholder
	}

	// compute outputs //
	// first layer
	for (int i = 0; i < net[0]->NumberOfOutputNodes; i++)
	{
		for (int j = 0; j < net[0]->NumberOfInputNodes - 1; j++)
		{
			outputs[0][i] += net[0]->weights[i][j] * point->Attributes[j];
		}
		outputs[0][i] += net[0]->weights[i][net[0]->NumberOfInputNodes - 1];
		outputs[0][i] = Sigmoid(outputs[0][i]);
	}
	// remaining layers
	for (int k = 1; k < config->NumberOfLayers; k++)
	{
		for (int i = 0; i < net[k]->NumberOfOutputNodes; i++)
		{
			for (int j = 0; j < net[k]->NumberOfInputNodes; j++)
			{
				outputs[k][i] += net[k]->weights[i][j] * outputs[k - 1][j];
			}
			//outputs[k][i] += net[k]->weights[i][net[k]->NumberOfInputNodes - 1];
			outputs[k][i] = Sigmoid(outputs[k][i]);
		}
	}

	return outputs;
}

double ANN::Sigmoid(double input)
{
	return 1.0 / (1 + exp(-input));
}

double ANN::ReturnPrediction(DataPoint* point, Config* config, ANNLayer** net)
{
	// return output of net, assuming last layer is 1-dimensional
	double** outputs = ReturnOutputs(point, config, net);
	return outputs[config->NumberOfLayers - 1][0];
}