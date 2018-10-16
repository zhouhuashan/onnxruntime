/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    convolve.cpp

Abstract:

    This module implements the convolution operation.

--*/

#include "mlasi.h"

//
// Define the number of working buffer elements required per thread.
//

#define MLAS_CONV_WORKING_BUFFER_SIZE_PER_THREAD \
    (MLAS_SGEMM_STRIDEN * MLAS_SGEMM_STRIDEK)

//
// Define the parameters to execute segments of a convolution operation on
// worker threads.
//

struct MLAS_CONV_WORK_BLOCK {
#if defined(MLAS_USE_WIN32_THREADPOOL)
    volatile LONG Counter;
    const MLAS_CONV_PARAMETERS* Parameters;
    const float* Input;
    const float* Filter;
    const float* Bias;
    float* WorkingBuffer;
    float* Output;
#endif
    struct SEGMENT {
        size_t StartN;
        size_t CountN;
    } Segments[MLAS_MAXIMUM_THREAD_COUNT];
};

void
MlasConvIm2Col(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    float* ColumnBuffer,
    size_t k,
    size_t CountK,
    size_t n,
    size_t CountN
    )
/*++

Routine Description:

    This routine converts the input image to a set of convolution patches
    appropriate for use with a GEMM operation.

    This implementation supports sampling a portion of the convolution
    patches. This avoids the need to allocate very large buffers to store
    all of the convolution patches at once, when the underyling GEMM
    implementation will already break up the operation into panels. Multiple
    threads can also be used to process different portions of the image.

Arguments:

    Parameters - Supplies the structure that contains the convolution
        parameters.

    Input - Supplies the input tensor.

    ColumnBuffer - Supplies the buffer to receive the convolution patches.

    k - Supplies the K to begin sampling the convolution patches.

    CountK - Supplies the count of K to sample for the convolution patches.

    n - Supplies the N to begin sampling the convolution patches.

    CountN - Supplies the count of N to sample for the convolution patches.

Return Value:

    None.

--*/
{
    constexpr size_t HeightShapeIndex = 0;
    constexpr size_t WidthShapeIndex = 1;

    size_t OutputWidth = Parameters->OutputShape[WidthShapeIndex];

    size_t StrideHeight = Parameters->StrideShape[HeightShapeIndex];
    size_t StrideWidth = Parameters->StrideShape[WidthShapeIndex];

    size_t nx = (n % OutputWidth);
    size_t ny = (n / OutputWidth);

    size_t OriginInputX = nx * StrideWidth;
    size_t OriginInputY = ny * StrideHeight;

    size_t OutputCountX = OutputWidth - nx;

    size_t InputHeight = Parameters->InputShape[HeightShapeIndex];
    size_t InputWidth = Parameters->InputShape[WidthShapeIndex];
    size_t InputArea = InputHeight * InputWidth;

    size_t KernelHeight = Parameters->KernelShape[HeightShapeIndex];
    size_t KernelWidth = Parameters->KernelShape[WidthShapeIndex];

    size_t kx = (k % KernelWidth);
    size_t ky = (k / KernelWidth) % KernelHeight;

    Input = Input + (k / (KernelHeight * KernelWidth)) * InputArea;

    size_t DilationHeight = Parameters->DilationShape[HeightShapeIndex];
    size_t DilationWidth = Parameters->DilationShape[WidthShapeIndex];

    size_t PaddingLeftY = Parameters->Padding[HeightShapeIndex];
    size_t PaddingLeftX = Parameters->Padding[WidthShapeIndex];

    for (size_t EndingK = k + CountK; k < EndingK; k++) {

        size_t CountX = OutputCountX;
        size_t InputY = (ky * DilationHeight) + OriginInputY - PaddingLeftY;
        size_t RowInitialInputX = (kx * DilationWidth) - PaddingLeftX;
        size_t InitialInputX = RowInitialInputX + OriginInputX;
        size_t RemainingN = CountN;

        do {

            if (CountX > RemainingN) {
                CountX = RemainingN;
            }

            RemainingN -= CountX;

            //
            // Check if the input is in the top/bottom padding region.
            //

            if (InputY < InputHeight) {

                size_t InputX = InitialInputX;
                const float* InputRow = &Input[InputY * InputWidth];

                do {

                    //
                    // Check if the input is in the left/right padding region.
                    //

                    if (InputX >= InputWidth) {

                        *ColumnBuffer++ = 0;
                        InputX += StrideWidth;
                        CountX--;

                    } else if (StrideWidth == 1) {

                        //
                        // Copy input elements to the column buffer.
                        //

                        size_t CountCopyX = InputWidth - InputX;

                        if (CountCopyX > CountX) {
                            CountCopyX = CountX;
                        }

                        CountX -= CountCopyX;

                        while (CountCopyX >= 4) {
                            MlasStoreFloat32x4(ColumnBuffer, MlasLoadFloat32x4(&InputRow[InputX]));
                            ColumnBuffer += 4;
                            InputX += 4;
                            CountCopyX -= 4;
                        }

                        while (CountCopyX > 0) {
                            *ColumnBuffer++ = InputRow[InputX++];
                            CountCopyX--;
                        }

                    } else if (InputX + CountX * StrideWidth <= InputWidth) {

                        do {
                            *ColumnBuffer++ = InputRow[InputX];
                            InputX += StrideWidth;
                        } while (--CountX > 0);

                    } else {

                        do {
                            *ColumnBuffer++ = (InputX < InputWidth) ? InputRow[InputX] : 0;
                            InputX += StrideWidth;
                        } while (--CountX > 0);
                    }

                } while (CountX > 0);

            } else {

                //
                // The entire input row is in the padding region.
                //

                MLAS_FLOAT32X4 ZeroFloat32x4 = MlasZeroFloat32x4();

                while (CountX >= 4) {
                    MlasStoreFloat32x4(ColumnBuffer, ZeroFloat32x4);
                    ColumnBuffer += 4;
                    CountX -= 4;
                }

                while (CountX > 0) {
                    MlasStoreFloat32(ColumnBuffer, ZeroFloat32x4);
                    ColumnBuffer++;
                    CountX--;
                }
            }

            CountX = OutputWidth;
            InputY += StrideHeight;
            InitialInputX = RowInitialInputX;

        } while (RemainingN > 0);

        //
        // Advance the kernel indices and advance to the next channel if the
        // entire kernel is complete.
        //

        if (++kx == KernelWidth) {

            if (++ky == KernelHeight) {

                Input += InputArea;

                ky = 0;
            }

            kx = 0;
        }
    }
}

void
MlasConvVol2Col(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    float* ColumnBuffer,
    size_t k,
    size_t CountK,
    size_t n,
    size_t CountN
    )
/*++

Routine Description:

    This routine converts the input volume to a set of convolution patches
    appropriate for use with a GEMM operation.

    This implementation supports sampling a portion of the convolution
    patches. This avoids the need to allocate very large buffers to store
    all of the convolution patches at once, when the underyling GEMM
    implementation will already break up the operation into panels. Multiple
    threads can also be used to process different portions of the image.

Arguments:

    Parameters - Supplies the structure that contains the convolution
        parameters.

    Input - Supplies the input tensor.

    ColumnBuffer - Supplies the buffer to receive the convolution patches.

    k - Supplies the K to begin sampling the convolution patches.

    CountK - Supplies the count of K to sample for the convolution patches.

    n - Supplies the N to begin sampling the convolution patches.

    CountN - Supplies the count of N to sample for the convolution patches.

Return Value:

    None.

--*/
{
    constexpr size_t DepthShapeIndex = 0;
    constexpr size_t HeightShapeIndex = 1;
    constexpr size_t WidthShapeIndex = 2;

    size_t OutputHeight = Parameters->OutputShape[HeightShapeIndex];
    size_t OutputWidth = Parameters->OutputShape[WidthShapeIndex];

    size_t StrideDepth = Parameters->StrideShape[DepthShapeIndex];
    size_t StrideHeight = Parameters->StrideShape[HeightShapeIndex];
    size_t StrideWidth = Parameters->StrideShape[WidthShapeIndex];

    size_t nx = (n % OutputWidth);
    size_t ny = ((n / OutputWidth) % OutputHeight);
    size_t nz = ((n / OutputWidth) / OutputHeight);

    size_t OutputCountX = OutputWidth - nx;
    size_t OutputCountY = OutputHeight - ny;

    size_t OriginInputX = nx * StrideWidth;
    size_t OriginInputY = ny * StrideHeight;
    size_t OriginInputZ = nz * StrideDepth;

    size_t InputDepth = Parameters->InputShape[DepthShapeIndex];
    size_t InputHeight = Parameters->InputShape[HeightShapeIndex];
    size_t InputWidth = Parameters->InputShape[WidthShapeIndex];
    size_t InputVolume = InputDepth * InputHeight * InputWidth;

    size_t KernelDepth = Parameters->KernelShape[DepthShapeIndex];
    size_t KernelHeight = Parameters->KernelShape[HeightShapeIndex];
    size_t KernelWidth = Parameters->KernelShape[WidthShapeIndex];

    size_t kx = (k % KernelWidth);
    size_t ky = (k / KernelWidth) % KernelHeight;
    size_t kz = ((k / KernelWidth) / KernelHeight) % KernelDepth;

    Input = Input + (k / (KernelDepth * KernelHeight * KernelWidth)) * InputVolume;

    size_t DilationDepth = Parameters->DilationShape[DepthShapeIndex];
    size_t DilationHeight = Parameters->DilationShape[HeightShapeIndex];
    size_t DilationWidth = Parameters->DilationShape[WidthShapeIndex];

    size_t PaddingLeftZ = Parameters->Padding[DepthShapeIndex];
    size_t PaddingLeftY = Parameters->Padding[HeightShapeIndex];
    size_t PaddingLeftX = Parameters->Padding[WidthShapeIndex];

    for (size_t EndingK = k + CountK; k < EndingK; k++) {

        size_t CountY = OutputCountY;
        size_t CountX = OutputCountX;
        size_t InputZ = (kz * DilationDepth) + OriginInputZ - PaddingLeftZ;
        size_t RowInitialInputY = (ky * DilationHeight) - PaddingLeftY;
        size_t InputY = RowInitialInputY + OriginInputY;
        size_t RowInitialInputX = (kx * DilationWidth) - PaddingLeftX;
        size_t InitialInputX = RowInitialInputX + OriginInputX;
        size_t RemainingN = CountN;

        do {

            if (CountX > RemainingN) {
                CountX = RemainingN;
            }

            RemainingN -= CountX;

            //
            // Check if the input is in the top/bottom or front/back padding region.
            //

            if (InputY < InputHeight && InputZ < InputDepth) {

                size_t InputX = InitialInputX;
                const float* InputRow = &Input[InputZ * (InputHeight * InputWidth) + InputY * InputWidth];

                do {

                    //
                    // Check if the input is in the left/right padding region.
                    //

                    if (InputX >= InputWidth) {

                        *ColumnBuffer++ = 0;
                        InputX += StrideWidth;
                        CountX--;

                    } else if (StrideWidth == 1) {

                        //
                        // Copy input elements to the column buffer.
                        //

                        size_t CountCopyX = InputWidth - InputX;

                        if (CountCopyX > CountX) {
                            CountCopyX = CountX;
                        }

                        CountX -= CountCopyX;

                        while (CountCopyX >= 4) {
                            MlasStoreFloat32x4(ColumnBuffer, MlasLoadFloat32x4(&InputRow[InputX]));
                            ColumnBuffer += 4;
                            InputX += 4;
                            CountCopyX -= 4;
                        }

                        while (CountCopyX > 0) {
                            *ColumnBuffer++ = InputRow[InputX++];
                            CountCopyX--;
                        }

                    } else if (InputX + CountX * StrideWidth <= InputWidth) {

                        do {
                            *ColumnBuffer++ = InputRow[InputX];
                            InputX += StrideWidth;
                        } while (--CountX > 0);

                    } else {

                        do {
                            *ColumnBuffer++ = (InputX < InputWidth) ? InputRow[InputX] : 0;
                            InputX += StrideWidth;
                        } while (--CountX > 0);
                    }

                } while (CountX > 0);

            } else {

                //
                // The entire input row is in the padding region.
                //

                MLAS_FLOAT32X4 ZeroFloat32x4 = MlasZeroFloat32x4();

                while (CountX >= 4) {
                    MlasStoreFloat32x4(ColumnBuffer, ZeroFloat32x4);
                    ColumnBuffer += 4;
                    CountX -= 4;
                }

                while (CountX > 0) {
                    MlasStoreFloat32(ColumnBuffer, ZeroFloat32x4);
                    ColumnBuffer++;
                    CountX--;
                }
            }

            CountX = OutputWidth;
            InputY += StrideHeight;
            InitialInputX = RowInitialInputX;

            if (--CountY == 0) {

                InputY = RowInitialInputY;
                InputZ += StrideDepth;

                CountY = OutputHeight;
            }

        } while (RemainingN > 0);

        //
        // Advance the kernel indices and advance to the next channel if the
        // entire kernel is complete.
        //

        if (++kx == KernelWidth) {

            if (++ky == KernelHeight) {

                if (++kz == KernelDepth) {

                    Input += InputVolume;

                    kz = 0;
                }

                ky = 0;
            }

            kx = 0;
        }
    }
}

void
MlasConvOperation(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    const float* Bias,
    float* ColumnBuffer,
    float* Output,
    size_t SegmentStartN,
    size_t SegmentCountN
    )
/*++

Routine Description:

    This routine implements the convolution operation.

Arguments:

    Parameters - Supplies the structure that contains the convolution
        parameters.

    Input - Supplies the input tensor.

    Filter - Supplies the filter tensor.

    Bias - Optionally supplies the bias vector.

    ColumnBuffer - Supplies the thread local slice of the working buffer.

    Output - Supplies the output tensor.

    SegmentStartN - Supplies the N to begin sampling the convolution patches.

    SegmentCountN - Supplies the count of N to sample for the convolution
        patches.

Return Value:

    None.

--*/
{
    size_t FilterCount = Parameters->FilterCount;
    size_t N = Parameters->N;
    size_t K = Parameters->K;

    //
    // Compute the strides to step through slices of the local segment.
    //
    // See MlasSgemmOperation.
    //

    uint32_t StrideN = MLAS_SGEMM_STRIDEN;
    uint32_t StrideK = MLAS_SGEMM_STRIDEK;

    if (SegmentCountN >= K) {

        while (StrideK / 2 >= K) {
            StrideN *= 2;
            StrideK /= 2;
        }

    } else {

        while (StrideN > 16 && StrideN / 2 >= SegmentCountN) {
            StrideK *= 2;
            StrideN /= 2;
        }
    }

    //
    // Step through each slice of the input tensor along the N dimension.
    //

    size_t CountN;

    for (size_t n = 0; n < SegmentCountN; n += CountN) {

        CountN = SegmentCountN - n;

        if (CountN > StrideN) {
            CountN = StrideN;
        }

        //
        // Step through each slice of the input tensor along the K dimension.
        //

        size_t CountK;
        float beta = 0.0f;
        float* SegmentOutput = Output + SegmentStartN + n;

        for (size_t k = 0; k < K; k += CountK) {

            CountK = K - k;

            if (CountK > StrideK) {
                CountK = StrideK;
            }

            if (Parameters->Dimensions == 2) {
                MlasConvIm2Col(Parameters, Input, ColumnBuffer, k, CountK,
                    SegmentStartN + n, CountN);
            } else {
                MlasConvVol2Col(Parameters, Input, ColumnBuffer, k, CountK,
                    SegmentStartN + n, CountN);
            }

            MlasSgemmOperation(CblasNoTrans, CblasNoTrans, FilterCount, CountN,
                CountK, 1.0f, Filter + k, K, ColumnBuffer, CountN, beta,
                SegmentOutput, N);

            beta = 1.0f;
        }

        //
        // Add the optional bias vector.
        //

        if (Bias != nullptr) {
            MlasBiasAdd(Bias, FilterCount, SegmentOutput, CountN, N);
        }
    }
}

#if defined(MLAS_USE_WIN32_THREADPOOL)

void
CALLBACK
MlasConvWorkCallback(
    PTP_CALLBACK_INSTANCE Instance,
    void* Context,
    PTP_WORK WorkObject
    )
/*++

Routine Description:

    This routine is invoked from a worker thread to execute a segment of a
    convolution operation.

Arguments:

    Instance - Supplies the callback instance object.

    Context - Supplies the pointer to the parameters for the SGEMM operation.

    WorkObject - Supplies the threadpool work object.

Return Value:

    None.

--*/
{
    UNREFERENCED_PARAMETER(Instance);
    UNREFERENCED_PARAMETER(WorkObject);

    MLAS_CONV_WORK_BLOCK* WorkBlock = (MLAS_CONV_WORK_BLOCK*)Context;

    LONG Index = InterlockedIncrement(&WorkBlock->Counter) - 1;

    MLAS_CONV_WORK_BLOCK::SEGMENT* Segment = &WorkBlock->Segments[Index];

    float* ColumnBuffer =
        WorkBlock->WorkingBuffer + Index * MLAS_CONV_WORKING_BUFFER_SIZE_PER_THREAD;

    MlasConvOperation(WorkBlock->Parameters, WorkBlock->Input, WorkBlock->Filter,
        WorkBlock->Bias, ColumnBuffer, WorkBlock->Output, Segment->StartN,
        Segment->CountN);
}

#endif

inline
bool
MlasConvTryMultithread(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    const float* Bias,
    float* WorkingBuffer,
    float* Output
    )
/*++

Routine Description:

    This routine attempts to launch a convolution operation across multiple
    threads.

Arguments:

    Parameters - Supplies the structure that contains the convolution
        parameters.

    Input - Supplies the input tensor.

    Filter - Supplies the filter tensor.

    Bias - Optionally supplies the bias vector.

    WorkingBuffer - Supplies a working buffer sized to the number of elements
        returned by MlasConvPrepare.

    Output - Supplies the output tensor.

Return Value:

    Returns true if the operation was completed across multiple threads, else
    false if the operation should fall back to a single thread.

--*/
{

#if defined(MLAS_USE_WIN32_THREADPOOL) || defined(MLAS_USE_OPENMP)

    MLAS_CONV_WORK_BLOCK WorkBlock;

    size_t N = Parameters->N;
    size_t ThreadStrideN = Parameters->ThreadStrideN;

    if (ThreadStrideN >= N) {
        return false;
    }

#if defined(MLAS_USE_WIN32_THREADPOOL)

    //
    // Create an object to submit work to the threadpool.
    //

    PTP_WORK WorkObject = CreateThreadpoolWork(MlasConvWorkCallback, &WorkBlock, nullptr);

    if (WorkObject == nullptr) {
        return false;
    }

    //
    // Initialize the common fields of the work block.
    //

    WorkBlock.Counter = 0;
    WorkBlock.Parameters = Parameters;
    WorkBlock.Input = Input;
    WorkBlock.Filter = Filter;
    WorkBlock.Bias = Bias;
    WorkBlock.WorkingBuffer = WorkingBuffer;
    WorkBlock.Output = Output;

#endif

    //
    // Segment the operation across multiple threads.
    //

    uint32_t Index = 0;
    size_t SegmentCountN;

    for (size_t SegmentStartN = 0; SegmentStartN < N; SegmentStartN += SegmentCountN) {

        SegmentCountN = N - SegmentStartN;

        if (SegmentCountN > ThreadStrideN) {
            SegmentCountN = ThreadStrideN;
        }

        WorkBlock.Segments[Index].StartN = SegmentStartN;
        WorkBlock.Segments[Index].CountN = SegmentCountN;

#if defined(MLAS_USE_WIN32_THREADPOOL)

        //
        // Execute one of the segments on a worker thread.
        //

        if (Index > 0) {
            SubmitThreadpoolWork(WorkObject);
        }

#endif

        Index++;
    }

#if defined(MLAS_USE_OPENMP)

    #pragma omp parallel num_threads(Index)
    {
        int tid = omp_get_thread_num();

        MLAS_CONV_WORK_BLOCK::SEGMENT* Segment = &WorkBlock.Segments[tid];

        float* ColumnBuffer =
            WorkingBuffer + tid * MLAS_CONV_WORKING_BUFFER_SIZE_PER_THREAD;

        MlasConvOperation(Parameters, Input, Filter, Bias, ColumnBuffer, Output,
            Segment->StartN, Segment->CountN);
    }

#elif defined(MLAS_USE_WIN32_THREADPOOL)

    //
    // Execute the remaining segment on this thread.
    //

    MlasConvWorkCallback(nullptr, &WorkBlock, WorkObject);

    //
    // Wait for the worker threads to complete.
    //

    WaitForThreadpoolWorkCallbacks(WorkObject, FALSE);
    CloseThreadpoolWork(WorkObject);

#endif

    return true;

#else

    //
    // No threading implementation is available.
    //

    return false;

#endif

}

void
MLASCALL
MlasConv(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    const float* Bias,
    float* WorkingBuffer,
    float* Output
    )
/*++

Routine Description:

    This routine implements the convolution operation.

Arguments:

    Parameters - Supplies the structure that contains the convolution
        parameters.

    Input - Supplies the input tensor.

    Filter - Supplies the filter tensor.

    Bias - Optionally supplies the bias vector.

    WorkingBuffer - Supplies a working buffer sized to the number of elements
        returned by MlasConvPrepare.

    Output - Supplies the output tensor.

Return Value:

    None.

--*/
{
    size_t FilterCount = Parameters->FilterCount;
    size_t N = Parameters->N;
    size_t K = Parameters->K;

    //
    // Evaluate how the convolution will be performed.
    //

    if (WorkingBuffer == nullptr) {

        //
        // Pointwise convolution can invoke GEMM directly with the input buffer.
        //

        MlasSgemm(CblasNoTrans, CblasNoTrans, FilterCount, N, K, 1.0f, Filter, K, Input, N, 0.0f, Output, N);

    } else if (FilterCount > N) {

        //
        // The filter count is larger than the output dimensions, so perform the
        // full matrix expansion and then invoke the threaded GEMM.
        //

        if (Parameters->Dimensions == 2) {
            MlasConvIm2Col(Parameters, Input, WorkingBuffer, 0, K, 0, N);
        } else {
            MlasConvVol2Col(Parameters, Input, WorkingBuffer, 0, K, 0, N);
        }

        MlasSgemm(CblasNoTrans, CblasNoTrans, FilterCount, N, K, 1.0f, Filter, K, WorkingBuffer, N, 0.0f, Output, N);

    } else {

        //
        // Attempt to launch the convolution across multiple threads or fall back
        // to a single thread.
        //

        if (!MlasConvTryMultithread(Parameters, Input, Filter, Bias, WorkingBuffer, Output)) {
            MlasConvOperation(Parameters, Input, Filter, Bias, WorkingBuffer, Output, 0, N);
        }

        return;
    }

    //
    // Add the optional bias vector.
    //

    if (Bias != nullptr) {
        MlasBiasAdd(Bias, FilterCount, Output, N, N);
    }
}

bool
MLASCALL
MlasConvPrepare(
    MLAS_CONV_PARAMETERS* Parameters,
    size_t Dimensions,
    int64_t InputChannels,
    const int64_t* InputShape,
    const int64_t* KernelShape,
    const int64_t* DilationShape,
    const int64_t* Padding,
    const int64_t* StrideShape,
    int64_t FilterCount,
    size_t* WorkingBufferSize
    )
/*++

Routine Description:

    This routine prepares for a convolution operation by computing required
    parameters including the required working buffer size for intermediate
    results.

Arguments:

    Parameters - Supplies the structure that stores the provided and computed
        parameters for the convolution operation.

    Dimensions - Supplies the number of dimensions (must be 2 or 3).

    InputChannels - Supplies the number of input channels.

    InputShape - Supplies the shape of the input tensor.

    KernelShape - Supplies the shape of the kernel transform.

    DilationShape - Supplies the shape of the dilation.

    PaddingShape - Supplies the number of zero padding elements at the edge of
        the input tensor.

    StrideShape - Supplies the shape of the stride.

    FilterCount - Supplies the number of rows of the filter matrix.

    WorkingBufferSize - Receives the number of elements to allocate for the
        working buffer for intermediate results.

Return Value:

    Returns true if implementation can support this operation.

--*/
{
    //
    // Support only 2D or 3D convolutions.
    //

    if (Dimensions != 2 && Dimensions != 3) {
        return false;
    }

    //
    // Save the convolution parameters.
    //

    Parameters->Dimensions = Dimensions;
    Parameters->InputChannels = size_t(InputChannels);
    Parameters->FilterCount = size_t(FilterCount);

    for (size_t dim = 0; dim < Dimensions; dim++) {
        Parameters->InputShape[dim] = size_t(InputShape[dim]);
        Parameters->KernelShape[dim] = size_t(KernelShape[dim]);
        Parameters->DilationShape[dim] = size_t(DilationShape[dim]);
        Parameters->Padding[dim * 2] = size_t(Padding[dim * 2]);
        Parameters->Padding[dim * 2 + 1] = size_t(Padding[dim * 2 + 1]);
        Parameters->StrideShape[dim] = size_t(StrideShape[dim]);
    }

    //
    // Compute the output shape.
    //

    for (size_t dim = 0; dim < Dimensions; dim++) {

        int64_t OutputShape = (InputShape[dim] + Padding[dim] + Padding[dim + Dimensions] -
            (DilationShape[dim] * (KernelShape[dim] - 1) + 1)) / StrideShape[dim] + 1;

        if (OutputShape <= 0) {
            return false;
        }

        Parameters->OutputShape[dim] = size_t(OutputShape);
    }

    //
    // Compute the GEMM N and K parameters.
    //

    size_t N = 1;
    size_t K = size_t(InputChannels);
    size_t Strides = 1;
    size_t Paddings = 0;

    for (size_t dim = 0; dim < Dimensions; dim++) {

        N *= Parameters->OutputShape[dim];
        K *= Parameters->KernelShape[dim];

        Strides *= Parameters->StrideShape[dim];
        Paddings |= (Parameters->Padding[dim] | Parameters->Padding[dim + Dimensions]);
    }

    Parameters->N = N;
    Parameters->K = K;

    //
    // Evaluate how the convolution will be performed.
    //

    if (K == size_t(InputChannels) && Strides == 1 && Paddings == 0) {

        //
        // Pointwise convolution can invoke GEMM directly with the input buffer.
        //

        *WorkingBufferSize = 0;

    } else if (size_t(FilterCount) > N) {

        //
        // The filter count is larger than the output dimensions, so perform the
        // full matrix expansion and then invoke the threaded GEMM.
        //

        *WorkingBufferSize = N * K;

    } else {

        //
        // Segment the operation across multiple threads by slicing the N
        // dimension (see MlasSgemmTryMultithread).
        //
        // Compute the number of target threads given the complexity of the
        // convolution operation. Small requests should run using the single
        // threaded path.
        //

        uint32_t TargetThreadCount;
        double Complexity = double(FilterCount) * double(N) * double(K);

        if (Complexity < double(MLAS_SGEMM_THREAD_COMPLEXITY * MLAS_MAXIMUM_THREAD_COUNT)) {
            TargetThreadCount = uint32_t(Complexity / double(MLAS_SGEMM_THREAD_COMPLEXITY)) + 1;
        } else {
            TargetThreadCount = MLAS_MAXIMUM_THREAD_COUNT;
        }

        uint32_t MaximumThreadCount = MlasPlatform.GetMaximumThreadCount();

        if (TargetThreadCount >= MaximumThreadCount) {
            TargetThreadCount = MaximumThreadCount;
        }

        //
        // Compute the thread stride for slicing the N dimension.
        //

        size_t StrideN = N / TargetThreadCount;

        if ((StrideN * TargetThreadCount) != N) {
            StrideN++;
        }

        if (TargetThreadCount > 1) {

            StrideN = (StrideN + MLAS_SGEMM_STRIDEN_THREAD_ALIGN - 1) & ~(MLAS_SGEMM_STRIDEN_THREAD_ALIGN - 1);

            if (StrideN >= N) {
                TargetThreadCount = 1;
            } else if (StrideN * (TargetThreadCount - 1) >= N) {
                TargetThreadCount--;
            }
        }

        Parameters->ThreadStrideN = StrideN;

        *WorkingBufferSize = TargetThreadCount * MLAS_CONV_WORKING_BUFFER_SIZE_PER_THREAD;
    }

    return true;
}
