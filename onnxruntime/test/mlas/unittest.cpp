/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    unittest.cpp

Abstract:

    This module implements unit tests of the MLAS library.

--*/

#include <stdio.h>
#include <mlas.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#endif

#if !defined(_countof)
#define _countof(_Array) (sizeof(_Array) / sizeof(_Array[0]))
#endif

class MatrixGuardBuffer
{
public:
    MatrixGuardBuffer(size_t Width, size_t Height, bool ReadOnly)
    {
        Construct(Width, Height, ReadOnly);
    }

    MatrixGuardBuffer(size_t MaximumDimension, bool ReadOnly)
    {
        Construct(MaximumDimension, MaximumDimension, ReadOnly);
    }

    ~MatrixGuardBuffer(void)
    {
#if defined(_WIN32)
        VirtualFree(_BaseBuffer, 0, MEM_RELEASE);
#else
        munmap(_BaseBuffer, _BaseBufferSize);
#endif
    }

    float* GetBuffer(size_t Elements)
    {
        return _GuardAddress - Elements;
    }

private:
    void Construct(size_t Width, size_t Height, bool ReadOnly)
    {
        const size_t PageSize = 4096;
        const size_t GuardPadding = 256 * 1024;

        size_t MatrixSize = Width * Height * sizeof(float);
        size_t AlignedMatrixSize = (MatrixSize + PageSize - 1) & ~(PageSize - 1);

        _BaseBufferSize = AlignedMatrixSize + GuardPadding;

#if defined(_WIN32)
        _BaseBuffer = VirtualAlloc(NULL, _BaseBufferSize, MEM_RESERVE, PAGE_NOACCESS);
        VirtualAlloc(_BaseBuffer, AlignedMatrixSize, MEM_COMMIT, PAGE_READWRITE);
#else
        _BaseBuffer = mmap(0, _BaseBufferSize, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        mprotect(_BaseBuffer, AlignedMatrixSize, PROT_READ | PROT_WRITE);
#endif

        float* GuardAddress = (float*)((unsigned char*)_BaseBuffer + AlignedMatrixSize);

        const int MinimumFillValue = -23;
        const int MaximumFillValue = 23;

        int FillValue = MinimumFillValue;
        float* FillAddress = (float*)((unsigned char*)GuardAddress - MatrixSize);

        while (FillAddress < GuardAddress) {

            *FillAddress++ = (float)FillValue;

            FillValue++;

            if (FillValue > MaximumFillValue) {
                FillValue = MinimumFillValue;
            }
        }

        if (ReadOnly) {
#if defined(_WIN32)
            DWORD OldProtect;
            VirtualProtect(_BaseBuffer, AlignedMatrixSize, PAGE_READONLY, &OldProtect);
#else
            mprotect(_BaseBuffer, AlignedMatrixSize, PROT_READ);
#endif
        }

        _GuardAddress = GuardAddress;
    }

private:
    void* _BaseBuffer;
    size_t _BaseBufferSize;
    float* _GuardAddress;
};

void
ReferenceSgemm(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    float alpha,
    const float* A,
    size_t lda,
    const float* B,
    size_t ldb,
    float beta,
    float* C,
    size_t ldc
    )
{
    if (TransA == CblasNoTrans) {

        if (TransB == CblasNoTrans) {

            for (size_t m = 0; m < M; m++) {

                for (size_t n = 0; n < N; n++) {

                    const float* a = A + (m * lda);
                    const float* b = B + n;
                    float* c = C + (m * ldc) + n;
                    float sum = 0.0f;

                    for (size_t k = 0; k < K; k++) {
                        sum += (*b * *a);
                        b += ldb;
                        a += 1;
                    }

                    *c = (*c * beta) + (sum * alpha);
                }
            }

        } else {

            for (size_t m = 0; m < M; m++) {

                for (size_t n = 0; n < N; n++) {

                    const float* a = A + (m * lda);
                    const float* b = B + (n * ldb);
                    float* c = C + (m * ldc) + n;
                    float sum = 0.0f;

                    for (size_t k = 0; k < K; k++) {
                        sum += (*b * *a);
                        b += 1;
                        a += 1;
                    }

                    *c = (*c * beta) + (sum * alpha);
                }
            }
        }

    } else {

        if (TransB == CblasNoTrans) {

            for (size_t m = 0; m < M; m++) {

                for (size_t n = 0; n < N; n++) {

                    const float* a = A + m;
                    const float* b = B + n;
                    float* c = C + (m * ldc) + n;
                    float sum = 0.0f;

                    for (size_t k = 0; k < K; k++) {
                        sum += (*b * *a);
                        b += ldb;
                        a += lda;
                    }

                    *c = (*c * beta) + (sum * alpha);
                }
            }

        } else {

            for (size_t m = 0; m < M; m++) {

                for (size_t n = 0; n < N; n++) {

                    const float* a = A + m;
                    const float* b = B + (n * ldb);
                    float* c = C + (m * ldc) + n;
                    float sum = 0.0f;

                    for (size_t k = 0; k < K; k++) {
                        sum += (*b * *a);
                        b += 1;
                        a += lda;
                    }

                    *c = (*c * beta) + (sum * alpha);
                }
            }
        }
    }
}

void
TrialSgemm(
    CBLAS_TRANSPOSE TransA,
    CBLAS_TRANSPOSE TransB,
    size_t M,
    size_t N,
    size_t K,
    float alpha,
    const float* A,
    size_t lda,
    const float* B,
    size_t ldb,
    float beta,
    float* C,
    float* CReference,
    size_t ldc
    )
{
    for (size_t f = 0; f < M * N; f++) {
        C[f] = -0.5f;
        CReference[f] = -0.5f;
    }

    MlasSgemm(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    ReferenceSgemm(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, CReference, ldc);

    for (size_t f = 0; f < M * N; f++) {
        // Sensitive to comparing positive/negative zero.
        if (C[f] != CReference[f]) {
            printf("mismatch TransA=%d, TransB=%d, M=%zd, N=%zd, K=%zd, alpha=%f, beta=%f!\n", TransA, TransB, M, N, K, alpha, beta);
        }
    }
}

void
TrialSgemm(
    size_t M,
    size_t N,
    size_t K,
    float alpha,
    MatrixGuardBuffer& BufferA,
    MatrixGuardBuffer& BufferB,
    float beta,
    MatrixGuardBuffer& BufferC,
    MatrixGuardBuffer& BufferCReference
    )
{
    const float* A = BufferA.GetBuffer(K * M);
    const float* B = BufferB.GetBuffer(N * K);
    float* C = BufferC.GetBuffer(N * M);
    float* CReference = BufferCReference.GetBuffer(N * M);

    TrialSgemm(CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, CReference, N);
    TrialSgemm(CblasNoTrans, CblasTrans, M, N, K, alpha, A, K, B, K, beta, C, CReference, N);
    TrialSgemm(CblasTrans, CblasNoTrans, M, N, K, alpha, A, M, B, N, beta, C, CReference, N);
    TrialSgemm(CblasTrans, CblasTrans, M, N, K, alpha, A, M, B, K, beta, C, CReference, N);
}

void
ExecuteSgemmTests(
    void
    )
{
    const size_t MaximumDimension = 320;

    MatrixGuardBuffer BufferA(MaximumDimension, true);
    MatrixGuardBuffer BufferB(MaximumDimension, true);
    MatrixGuardBuffer BufferC(MaximumDimension, false);
    MatrixGuardBuffer BufferCReference(MaximumDimension, false);

    // Trial balloons.
    for (size_t b = 1; b < 16; b++) {
        TrialSgemm(b, b, b, 1.0f, BufferA, BufferB, 0.0f, BufferC, BufferCReference);
    }
    for (size_t b = 16; b <= 256; b <<= 1) {
        TrialSgemm(b, b, b, 1.0f, BufferA, BufferB, 0.0f, BufferC, BufferCReference);
    }
    for (size_t b = 256; b < 320; b += 32) {
        TrialSgemm(b, b, b, 1.0f, BufferA, BufferB, 0.0f, BufferC, BufferCReference);
    }

    static const float multipliers[] = { 0.0f, -0.0f, 0.25f, -0.5f, 1.0f, -1.0f };

    for (size_t N = 1; N < 128; N++) {
        for (size_t K = 1; K < 128; K++) {
            for (size_t a = 0; a < _countof(multipliers); a++) {
                for (size_t b = 0; b < _countof(multipliers); b++) {
                    TrialSgemm(1, N, K, multipliers[a], BufferA, BufferB, multipliers[b], BufferC, BufferCReference);
                }
            }
        }
    }

    for (size_t a = 0; a < _countof(multipliers); a++) {
        float alpha = multipliers[a];

        for (size_t b = 0; b < _countof(multipliers); b++) {
            float beta = multipliers[b];

            for (size_t M = 16; M < 160; M += 32) {
                for (size_t N = 16; N < 160; N += 32) {

                    static const size_t ks[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 32, 48, 64, 118, 119, 120, 121, 122, 160, 240, 320 };
                    for (size_t k = 0; k < _countof(ks); k++) {
                        size_t K = ks[k];

                        TrialSgemm(M, N, K, alpha, BufferA, BufferB, beta, BufferC, BufferCReference);
                        TrialSgemm(M + 1, N, K, alpha, BufferA, BufferB, beta, BufferC, BufferCReference);
                        TrialSgemm(M, N + 1, K, alpha, BufferA, BufferB, beta, BufferC, BufferCReference);
                        TrialSgemm(M + 1, N + 1, K, alpha, BufferA, BufferB, beta, BufferC, BufferCReference);
                        TrialSgemm(M + 3, N + 2, K, alpha, BufferA, BufferB, beta, BufferC, BufferCReference);
                        TrialSgemm(M + 4, N, K, alpha, BufferA, BufferB, beta, BufferC, BufferCReference);
                        TrialSgemm(M, N + 4, K, alpha, BufferA, BufferB, beta, BufferC, BufferCReference);
                        TrialSgemm(M + 4, N + 4, K, alpha, BufferA, BufferB, beta, BufferC, BufferCReference);
                        TrialSgemm(M + 3, N + 7, K, alpha, BufferA, BufferB, beta, BufferC, BufferCReference);
                        TrialSgemm(M + 8, N, K, alpha, BufferA, BufferB, beta, BufferC, BufferCReference);
                        TrialSgemm(M, N + 8, K, alpha, BufferA, BufferB, beta, BufferC, BufferCReference);
                        TrialSgemm(M + 12, N + 12, K, alpha, BufferA, BufferB, beta, BufferC, BufferCReference);
                        TrialSgemm(M + 13, N, K, alpha, BufferA, BufferB, beta, BufferC, BufferCReference);
                        TrialSgemm(M, N + 15, K, alpha, BufferA, BufferB, beta, BufferC, BufferCReference);
                        TrialSgemm(M + 15, N + 15, K, alpha, BufferA, BufferB, beta, BufferC, BufferCReference);
                    }
                }
                printf("a %zd/%zd b %zd/%zd M %zd\n", a, _countof(multipliers), b, _countof(multipliers), M);
            }
        }
    }

    for (size_t M = 1; M < 160; M++) {
        for (size_t N = 1; N < 160; N++) {
            for (size_t K = 1; K < 160; K++) {
                TrialSgemm(M, N, K, 1.0f, BufferA, BufferB, 0.0f, BufferC, BufferCReference);
            }
        }
        printf("M %zd\n", M);
    }

    for (size_t M = 160; M < 320; M += 24) {
        for (size_t N = 112; N < 320; N += 24) {
            for (size_t K = 1; K < 16; K++) {
                TrialSgemm(M, N, K, 1.0f, BufferA, BufferB, 0.0f, BufferC, BufferCReference);
            }
            for (size_t K = 16; K < 160; K += 32) {
                TrialSgemm(M, N, K, 1.0f, BufferA, BufferB, 0.0f, BufferC, BufferCReference);
            }
        }
        printf("M %zd\n", M);
    }
}

#if 0
#if defined(_WIN32)

extern uint32_t MlasMaximumThreadCount;

void
EvaluateThreadingPerformance(
    void
    )
{
    SYSTEM_INFO SystemInfo;

    GetSystemInfo(&SystemInfo);

    const size_t MaximumDimension = 4096;

    MatrixGuardBuffer BufferA(MaximumDimension, true);
    MatrixGuardBuffer BufferB(MaximumDimension, true);
    MatrixGuardBuffer BufferC(MaximumDimension, false);

    for (size_t M = 16; M <= MaximumDimension; M <<= 1) {
        for (size_t N = 16; N <= MaximumDimension; N <<= 1) {
            for (size_t K = 16; K <= MaximumDimension; K <<= 1) {

                const float* A = BufferA.GetBuffer(K * M);
                const float* B = BufferB.GetBuffer(N * K);
                float* C = BufferC.GetBuffer(N * M);

                //
                // Compute the number of iterations to run for at least five seconds.
                //

                MlasMaximumThreadCount = 1;

                ULONG64 NumberIterations = 0;
                DWORD start = GetTickCount();
                DWORD stop;
                do {
                    MlasSgemm(CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
                    stop = GetTickCount();
                    NumberIterations++;
                } while ((stop - start) <= 5000);

                //
                // Determine the performance for a range of thread counts.
                //

                static DWORD cpus[] = { 1, 2, 4, 8, 12 };
                for (size_t cpu = 0; cpu < _countof(cpus); cpu++) {

                    if (cpus[cpu] <= SystemInfo.dwNumberOfProcessors) {
                        MlasMaximumThreadCount = cpus[cpu];
                    } else {
                        break;
                    }

                    start = GetTickCount();
                    for (size_t iters = 0; iters < NumberIterations; iters++) {
                        MlasSgemm(CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
                        stop = GetTickCount();
                        if ((stop - start) > 20000) {
                            break;
                        }
                    }

                    printf("%zd,%zd,%zd cpus=%d, iters=%I64d, time=%d %c\n", M, N, K,
                        MlasMaximumThreadCount, NumberIterations, stop - start, ((stop - start) > 5100) ? '!' : '\0');
                }

                fflush(stdout);
            }
        }
    }
}

#endif
#endif

int
#if defined(_WIN32)
__cdecl
#endif
main(
    void
    )
{
    ExecuteSgemmTests();
//    EvaluateThreadingPerformance();

    return 0;
}
