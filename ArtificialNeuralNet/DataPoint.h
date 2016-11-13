#pragma once
# include "Config.h"

struct DataPoint
{
	DataPoint()
	{
		// constructor
	}

	~DataPoint()
	{
		// destructor
		if (Attributes) delete[] Attributes;
	}

	// Ground truth class value
	short ClassName;
	// Store values of continuous attributes as doubles
	double* Attributes;
};
