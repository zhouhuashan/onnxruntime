//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
// File: lut.cpp
// <OWNER>dl-optimization</OWNER>
// http://aka.ms/dl-optimization
//------------------------------------------------------------------------------

#include "stdafx.h"
#include "lut.h"
#include <math.h>

bool LUT::Sigmoid2TabInit() {
    int size = 30000000;
    sigmoid2tab = new double[size];
    int i = 0;
    // x larger than 15 or smaller than -15 get staurated from sigmoid.		
    // We can certainly increase the accuracy by making x take finer steps. However, that would increase the size of the lookup table.		
    for (double x = -15.0f; x < 15.0f; x += 0.000001f)
    {
        sigmoid2tab[i++] = (1 / (1 + exp(-x)));
    }

    return true;
}

bool LUT::Tanh2TabInit() {
    int size = 30000000;
    tanh2tab = new double[size];
    int i = 0;
    for (double x = -15.0f; x < 15.0f; x += 0.000001f)
    {
        tanh2tab[i++] = tanh(x);
    }

    return true;
}

LUT::LUT() {
	this->Sigmoid2TabInit();
	this->Tanh2TabInit();
}

LUT::~LUT() {
    // TODO: delete arrays allocated for the singleton _inst.
}