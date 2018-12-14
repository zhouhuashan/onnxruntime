#include "InnerUtility.h"

//template <typename T, typename K>
//void dim_flat(std::vector<T> const & v, std::vector<K>& output)
//{
//	for (int i = 0; i < v.size(); i++)
//	{
//		output.push_back((K)v[i]);
//	}
//}
//
//template <typename T, typename K>
//size_t dim_flat(std::vector<T> const & v, K* output)
//{
//	auto pCurrent = output;
//	for (int i = 0; i < v.size(); i++)
//	{
//		output[i] = (K)v[i];
//	}
//	return v.size();;
//}
//
//template <typename T, typename K>
//void dim_flat(std::vector<std::vector<T>> const & v, std::vector<K>& output)
//{
//	for (int i = 0; i < v.size(); i++)
//	{
//		dim_flat(v[i], output);
//	}
//
//}
//
//template <typename T, typename K>
//size_t dim_flat(std::vector<std::vector<T>> const & v, K* output)
//{
//	auto pCurrent = output;
//	
//	for (int i = 0; i < v.size(); i++)
//	{
//		auto len = dim_flat(v[i], pCurrent);
//		pCurrent += len;
//	}
//
//	return pCurrent - output;
//};