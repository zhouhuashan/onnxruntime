#pragma once
#include <vector>
#include <memory>
#define USE_MKL

#ifdef USE_MKL
#include <mkl.h>
#define MKL_ALIGN 64
#endif

template <typename T, typename K>
void dim_flat(std::vector<T> const & v, std::vector<K>& output)
{
	for (int i = 0; i < v.size(); i++)
	{
		output.push_back((K)v[i]);
	}
}

template <typename T, typename K>
size_t dim_flat(std::vector<T> const & v, K* output)
{
	auto pCurrent = output;
	for (int i = 0; i < v.size(); i++)
	{
		output[i] = (K)v[i];
	}
	return v.size();;
}

template <typename T, typename K>
void dim_flat(std::vector<std::vector<T>> const & v, std::vector<K>& output)
{
	for (int i = 0; i < v.size(); i++)
	{
		dim_flat(v[i], output);
	}

}

template <typename T, typename K>
size_t dim_flat(std::vector<std::vector<T>> const & v, K* output)
{
	auto pCurrent = output;
	auto sub_size = 0;
	for (int i = 0; i < v.size(); i++)
	{
		auto len = dim_flat(v[i], pCurrent);
		if (i == 0)
		{
			sub_size = len;
		}
		assert(sub_size == len);
		pCurrent += len;
	}

	return pCurrent - output;
};

#ifdef USE_MKL
template<typename T>
class memoryDeleter
{
public:
	void operator()(T *p)
	{
		mkl_free(p);
	}
};
#else
template<typename T>
class memoryDeleter
{
public:
	void operator()(T *p)
	{
		free(p);
	}
};
#endif

template<typename T>
inline std::shared_ptr<T> MemAllocate(size_t size)
{
#ifdef USE_MKL
	return std::shared_ptr<T>((T*)mkl_calloc(size, sizeof(T), MKL_ALIGN), memoryDeleter<T>());
#else
	return std::shared_ptr<T>(calloc(size, sizeof(T)), memoryDeleter);
#endif
}

inline const std::string PathCombine(std::string  folder, std::string  fileName)
{
	return folder + "\\" + fileName;
}