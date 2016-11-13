#pragma once

struct ANNLayer
{
	ANNLayer()
	{
		// constructor
	}

	~ANNLayer()
	{
		// destructor
		for (int j = 0; j < NumberOfInputNodes; j++)
		{
			delete[] weights[j];
		}
	}

	int NumberOfInputNodes;
	int NumberOfOutputNodes;
	double** weights;
};