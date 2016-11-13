#pragma once
#include "DataPoint.h"
#include "Config.h"

class DataHandler
{
public:
	DataHandler();
	~DataHandler();

	int NumberOfDataPoints;
	short DataType;
	short NumberOfAttributes;
	short* NumberOfAttributeValues;
	DataPoint** DataPoints;
};

