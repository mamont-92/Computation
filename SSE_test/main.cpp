#include <QCoreApplication>
#include <chrono>
#include <QDebug>

#include <malloc.h>
#include "xmmintrin.h"



void min_max_sse(float * alligned_ptr, size_t num, float & min, float & max)
{
    float res_min, res_max;
    __m128 * ptr = (__m128*) alligned_ptr;
    __m128 min_val = ptr[0];
    __m128 max_val = min_val;

    for(size_t i = 0; i < num/4; ++i){
        __m128 val = ptr[i];
        min_val = _mm_min_ps(min_val, val);
        max_val = _mm_max_ps(max_val, val);
    }

    for (int i = 0; i < 3; i++) {
        min_val = _mm_min_ps(min_val, _mm_shuffle_ps(min_val, min_val, 0x93));
        max_val = _mm_max_ps(max_val, _mm_shuffle_ps(max_val, max_val, 0x93));
    }
    _mm_store_ss(&res_min, min_val);
    _mm_store_ss(&res_max, max_val);

    min = res_min;
    max = res_max;
}

void min_max_scalar_if(float * ptr, size_t num, float & min, float & max)
{
    float res_min = ptr[0];
    float res_max = res_min;

    for(size_t i = 0; i < num; ++i){
        float val = ptr[i];
        if(val < res_min)
            res_min = val;
        if (val > res_max)
            res_max = val;
    }

    min = res_min;
    max = res_max;
}

void min_max_scalar_if_openmp(float * ptr, size_t num, float & min, float & max)
{
    float res_min = ptr[0];
    float res_max = res_min;

#pragma omp parallel
    {
        float localMin = res_min;
        float localMax = res_max;

#pragma omp for nowait
        for(qint64 i = 0; i <num; ++i){
            float val = ptr[i];
            if(val < localMin)
                localMin = val;
            if (val > localMax)
                localMax = val;
        }

#pragma omp critical
        {
            if(res_min > localMin)
                res_min = localMin;
            if(res_max < localMax)
                res_max = localMax;
        }
    }

    min = res_min;
    max = res_max;
}


void min_max_scalar_if_else(float * ptr, size_t num, float & min, float & max)
{
    float res_min = ptr[0];
    float res_max = res_min;

    for(size_t i = 0; i < num; ++i){
        float val = ptr[i];
        if(val < res_min)
            res_min = val;
        else if(val > res_max)
            res_max = val;
    }

    min = res_min;
    max = res_max;
}

template<typename Func>
void callMinWithTime(Func func, float * ptr, size_t num, float & min, float & max){
    auto start = std::chrono::system_clock::now();
    func(ptr, num, min, max);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;

    qDebug() << "min: " << min << "max: " << max << "elapsed: " << elapsed_seconds.count();
}

int main(int argc, char *argv[])
{
    qDebug() << "hello SSE";

    auto alignment = 16;
    auto N = 4*100000000;

    float *  ptr = (float*)_aligned_malloc(N*sizeof(float), alignment);
    if (!ptr)
        qDebug() << "mem alloc error";

    for(int i = 0; i < N; ++i){
        ptr[i] = (float)i;
    }

    float min, max;
    callMinWithTime(min_max_scalar_if, ptr, N, min, max);
    callMinWithTime(min_max_scalar_if_else, ptr, N, min, max);
    callMinWithTime(min_max_sse, ptr, N, min, max);
    callMinWithTime(min_max_scalar_if_openmp, ptr, N, min, max);

    return 0;
}
