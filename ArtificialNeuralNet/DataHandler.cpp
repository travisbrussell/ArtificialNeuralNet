#include "stdafx.h"
#include "DataHandler.h"
#include <time.h>
#include <stdlib.h>
#include <random>

DataHandler::DataHandler()
{
}

DataHandler::~DataHandler()
{
	delete[] DataPoints;
}

