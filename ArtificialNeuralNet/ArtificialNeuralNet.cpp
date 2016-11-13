// ArtificialNeuralNet.cpp : main project file.

#include "stdafx.h"
#include "ANN.h"
#include "DataHandler.h"
#include <string>
#include <fstream>
using namespace System;

void ANNTest();
void discreteAttributeTest(std::string trainAddress);
void continuousAttributeTest(std::string trainAddress);
DataHandler* ReturnCongressData(Config* config, std::string trainAddress);
DataHandler* ReturnPimaData(Config* config, std::string trainAddress);

int main(array<System::String ^> ^args)
{
    Console::WriteLine(L"Artificial neural network test");
	// Learn UCI Congress Voting Records data set
	std::string Congress = "C:\\Users\\TRussell\\Desktop\\ML Notes\\HW 1\\data\\Congress\\house-votes-84.txt";
	discreteAttributeTest(Congress);
	// Learn UCI Pima Indians Diabetes data set
	std::string Pima = "C:\\Users\\TRussell\\Desktop\\ML Notes\\HW 1\\data\\Pima\\pima-indians-diabetes.data";
	continuousAttributeTest(Pima);
    return 0;
}

void discreteAttributeTest(std::string trainAddress)
{
	Console::WriteLine(L"Discrete attribute test: ");

	// set up config
	// Congress parameters
	Config* config = new Config();
	config->LearningRate = 0.5;
	config->NumberOfLayers = 3;
	config->LayerSizes = new int[config->NumberOfLayers];
	config->LayerSizes[0] = 2;
	config->LayerSizes[1] = 2;
	config->LayerSizes[2] = 1;
	config->NumberOfAttributes = 16;
	config->NumberOfEpics = 100;
	config->NumberOfTrainingPoints = 200;
	config->NumberOfTestingPoints = 100;

	// read data
	DataHandler* AllData = ReturnCongressData(config, trainAddress);

	// prep training data
	DataPoint** trainingData = new DataPoint*[config->NumberOfTrainingPoints];
	for (int j = 0; j < config->NumberOfTrainingPoints; j++)
	{
		trainingData[j] = new DataPoint();
		trainingData[j]->ClassName = AllData->DataPoints[j]->ClassName;
		trainingData[j]->Attributes = new double[config->NumberOfAttributes + 1];
		for (int i = 0; i < config->NumberOfAttributes; i++)
		{
			trainingData[j]->Attributes[i] = AllData->DataPoints[j]->Attributes[i];
		}
		trainingData[j]->Attributes[config->NumberOfAttributes] = 1;  // last attribute is constant for threshold
	}

	// prep net
	ANNLayer** net = new ANNLayer*[config->NumberOfLayers];
	net[0] = new ANNLayer();
	net[0]->NumberOfInputNodes = config->NumberOfAttributes + 1;
	for (int j = 1; j < config->NumberOfLayers; j++)
	{
		net[j] = new ANNLayer();
		net[j - 1]->NumberOfOutputNodes = config->LayerSizes[j];
		net[j]->NumberOfInputNodes = config->LayerSizes[j] + 1;
	}
	net[config->NumberOfLayers - 1]->NumberOfOutputNodes = 1;

	// set weights
	for (int k = 0; k < config->NumberOfLayers; k++)
	{
		net[k]->weights = new double*[net[k]->NumberOfOutputNodes];
		for (int i = 0; i < net[k]->NumberOfOutputNodes; i++)
		{
			net[k]->weights[i] = new double[net[k]->NumberOfInputNodes];
			for (int j = 0; j < net[k]->NumberOfInputNodes; j++)
			{
				net[k]->weights[i][j] = 0.1;
			}
		}
	}

	// train net
	ANN* engine = new ANN();
	engine->TrainANN(net, config, trainingData);

	// prep testing data
	DataPoint** testingData = new DataPoint*[config->NumberOfTestingPoints];
	int offset = config->NumberOfTrainingPoints;
	for (int j = 0; j < config->NumberOfTestingPoints; j++)
	{
		testingData[j] = new DataPoint();
		testingData[j]->ClassName = AllData->DataPoints[offset + j]->ClassName;
		testingData[j]->Attributes = new double[config->NumberOfAttributes + 1];
		for (int i = 0; i < config->NumberOfAttributes; i++)
		{
			testingData[j]->Attributes[i] = AllData->DataPoints[offset + j]->Attributes[i];
		}
		testingData[j]->Attributes[config->NumberOfAttributes] = 1;  // last attribute is constant for threshold
	}

	// test net
	int TP = 0;
	int FP = 0;
	int TN = 0;
	int FN = 0;
	for (int j = 0; j < config->NumberOfTestingPoints; j++)
	{
		DataPoint* point = testingData[j];
		double numPrediction = engine->ReturnPrediction(point, config, net);
		bool prediction = numPrediction >= 0.5;
		bool groundTruth = point->ClassName > 0;
		if (prediction && groundTruth)
		{
			TP++;
		}
		if (prediction && !groundTruth)
		{
			FP++;
		}
		if (!prediction && groundTruth)
		{
			FN++;
		}
		if (!prediction && !groundTruth)
		{
			TN++;
		}
	}
	double Accuracy = (double)(TP + TN) / (config->NumberOfTestingPoints);
	Console::WriteLine("Accuracy is " + Accuracy);
	Console::ReadLine();
}

void continuousAttributeTest(std::string trainAddress)
{
	Console::WriteLine(L"Continuous attribute test: ");

	// set up config
	// pima parameters
	Config* config = new Config();
	config->LearningRate = 0.075;
	config->NumberOfLayers = 3;
	config->LayerSizes = new int[config->NumberOfLayers];
	config->LayerSizes[0] = 3;
	config->LayerSizes[1] = 3;
	config->LayerSizes[2] = 1;
	config->NumberOfAttributes = 8;
	config->NumberOfEpics = 1000;
	config->NumberOfTrainingPoints = 500;
	config->NumberOfTestingPoints = 100;

	// read data
	DataHandler* AllData = ReturnPimaData(config, trainAddress);

	// prep training data
	DataPoint** trainingData = new DataPoint*[config->NumberOfTrainingPoints];
	for (int j = 0; j < config->NumberOfTrainingPoints; j++)
	{
		trainingData[j] = new DataPoint();
		trainingData[j]->ClassName = AllData->DataPoints[j]->ClassName;
		trainingData[j]->Attributes = new double[config->NumberOfAttributes + 1];
		for (int i = 0; i < config->NumberOfAttributes; i++)
		{
			trainingData[j]->Attributes[i] = AllData->DataPoints[j]->Attributes[i];
		}
		trainingData[j]->Attributes[config->NumberOfAttributes] = 1;  // last attribute is constant for threshold
	}

	// prep net
	ANNLayer** net = new ANNLayer*[config->NumberOfLayers];
	net[0] = new ANNLayer();
	net[0]->NumberOfInputNodes = config->NumberOfAttributes + 1;
	for (int j = 1; j < config->NumberOfLayers; j++)
	{
		net[j] = new ANNLayer();
		net[j - 1]->NumberOfOutputNodes = config->LayerSizes[j];
		net[j]->NumberOfInputNodes = config->LayerSizes[j] + 1;
	}
	net[config->NumberOfLayers - 1]->NumberOfOutputNodes = 1;

	// set weights
	for (int k = 0; k < config->NumberOfLayers; k++)
	{
		net[k]->weights = new double*[net[k]->NumberOfOutputNodes];
		for (int i = 0; i < net[k]->NumberOfOutputNodes; i++)
		{
			net[k]->weights[i] = new double[net[k]->NumberOfInputNodes];
			for (int j = 0; j < net[k]->NumberOfInputNodes; j++)
			{
				net[k]->weights[i][j] = 0.1;
			}
		}
	}

	// train net
	ANN* engine = new ANN();
	engine->TrainANN(net, config, trainingData);

	// prep testing data
	DataPoint** testingData = new DataPoint*[config->NumberOfTestingPoints];
	int offset = config->NumberOfTrainingPoints;
	for (int j = 0; j < config->NumberOfTestingPoints; j++)
	{
		testingData[j] = new DataPoint();
		testingData[j]->ClassName = AllData->DataPoints[offset + j]->ClassName;
		testingData[j]->Attributes = new double[config->NumberOfAttributes + 1];
		for (int i = 0; i < config->NumberOfAttributes; i++)
		{
			testingData[j]->Attributes[i] = AllData->DataPoints[offset + j]->Attributes[i];
		}
		testingData[j]->Attributes[config->NumberOfAttributes] = 1;  // last attribute is constant for threshold
	}

	// test net
	int TP = 0;
	int FP = 0;
	int TN = 0;
	int FN = 0;
	for (int j = 0; j < config->NumberOfTestingPoints; j++)
	{
		DataPoint* point = testingData[j];
		double numPrediction = engine->ReturnPrediction(point, config, net);
		bool prediction = numPrediction >= 0.5;
		bool groundTruth = point->ClassName > 0;
		if (prediction && groundTruth)
		{
			TP++;
		}
		if (prediction && !groundTruth)
		{
			FP++;
		}
		if (!prediction && groundTruth)
		{
			FN++;
		}
		if (!prediction && !groundTruth)
		{
			TN++;
		}
	}
	double Accuracy = (double)(TP + TN) / (config->NumberOfTestingPoints);
	Console::WriteLine("Accuracy is " + Accuracy);
	Console::ReadLine();
}

DataHandler* ReturnCongressData(Config* config, std::string trainAddress)
{
	int numberOfDataPoints = 435;

	// set up data handler
	DataHandler* dataHandler = new DataHandler();
	dataHandler->NumberOfDataPoints = numberOfDataPoints;
	dataHandler->DataType = 1;
	dataHandler->DataPoints = new DataPoint*[dataHandler->NumberOfDataPoints];

	std::ifstream congressDataFile(trainAddress);
	for (int j = 0; j < numberOfDataPoints; j++)
	{
		dataHandler->DataPoints[j] = new DataPoint();
		std::string className;
		std::getline(congressDataFile, className, ',');
		if (className == "republican")
		{
			dataHandler->DataPoints[j]->ClassName = 0;
		}
		else
		{
			dataHandler->DataPoints[j]->ClassName = 1;
		}
		dataHandler->DataPoints[j]->Attributes = new double[config->NumberOfAttributes];
		for (int attributeIndex = 0; attributeIndex < config->NumberOfAttributes - 1; attributeIndex++)
		{
			std::string attribute;
			std::getline(congressDataFile, attribute, ',');
			if (attribute == "y")
			{
				dataHandler->DataPoints[j]->Attributes[attributeIndex] = 1;
			}
			if (attribute == "n")
			{
				dataHandler->DataPoints[j]->Attributes[attributeIndex] = 2;
			}
			if (attribute == "?")
			{
				dataHandler->DataPoints[j]->Attributes[attributeIndex] = 3;
			}
		}
		// final attribute
		std::string lastAttribute;
		std::getline(congressDataFile, lastAttribute, '\n');
		int index = config->NumberOfAttributes - 1;
		if (lastAttribute == "y")
		{
			dataHandler->DataPoints[j]->Attributes[index] = 1;
		}
		if (lastAttribute == "n")
		{
			dataHandler->DataPoints[j]->Attributes[index] = 2;
		}
		if (lastAttribute == "?")
		{
			dataHandler->DataPoints[j]->Attributes[index] = 3;
		}
	}
	// close input stream
	congressDataFile.close();

	return dataHandler;
}

DataHandler* ReturnPimaData(Config* config, std::string trainAddress)
{
	int numberOfDataPoints = 768;

	// set up data handler
	DataHandler* dataHandler = new DataHandler();
	dataHandler->NumberOfDataPoints = numberOfDataPoints;
	dataHandler->DataType = 2;
	dataHandler->DataPoints = new DataPoint*[dataHandler->NumberOfDataPoints];

	std::ifstream pimaDataFile(trainAddress);
	for (int j = 0; j < numberOfDataPoints; j++)
	{
		dataHandler->DataPoints[j] = new DataPoint();
		dataHandler->DataPoints[j]->Attributes = new double[config->NumberOfAttributes];
		for (int attributeIndex = 0; attributeIndex < config->NumberOfAttributes; attributeIndex++)
		{
			std::string attribute;
			std::getline(pimaDataFile, attribute, ',');
			dataHandler->DataPoints[j]->Attributes[attributeIndex] = std::stod(attribute) / 200;
		}
		// class value
		std::string lastAttribute;
		std::getline(pimaDataFile, lastAttribute, '\n');
		int index = config->NumberOfAttributes;
		if (lastAttribute == "0")
		{
			dataHandler->DataPoints[j]->ClassName = 0;
		}
		else
		{
			dataHandler->DataPoints[j]->ClassName = 1;
		}
	}
	// close input stream
	pimaDataFile.close();

	//mine max and min values from training data
	config->MinContinuousValues = new double[config->NumberOfAttributes];
	config->MaxContinuousValues = new double[config->NumberOfAttributes];

	// initialize min and max values based on first data point
	for (int k = 0; k < config->NumberOfAttributes; k++)
	{
		config->MinContinuousValues[k] = dataHandler->DataPoints[0]->Attributes[k];
		config->MaxContinuousValues[k] = dataHandler->DataPoints[0]->Attributes[k];
	}

	// find max and min attribute values from training points
	for (int j = 1; j < config->NumberOfTrainingPoints; j++)
	{
		for (int k = 0; k < config->NumberOfAttributes; k++)
		{
			double value = dataHandler->DataPoints[j]->Attributes[k];
			if (value < config->MinContinuousValues[k])
			{
				config->MinContinuousValues[k] = value;
			}
			if (value > config->MaxContinuousValues[k])
			{
				config->MaxContinuousValues[k] = value;
			}
		}

	}

	return dataHandler;
}
