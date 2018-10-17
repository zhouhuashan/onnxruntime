// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#ifdef USE_CUDA
#include "core/providers/cuda/cuda_provider_factory.h"
#endif
#ifdef USE_MKLDNN
#include "core/providers/mkldnn/mkldnn_provider_factory.h"
#endif