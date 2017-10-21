#ifndef LOTUSIR_CORE_GRAPH_TENSORUTILS_H
#define LOTUSIR_CORE_GRAPH_TENSORUTILS_H

#include <vector>

#include "core/protobuf/graph.pb.h"
#include "status.h"

namespace Lotus
{
    namespace Utils
    {
        class TensorUtils
        {
        public:

            static Common::Status UnpackTensor(const LotusIR::TensorProto& p_tensor, /*out*/ std::vector<std::string>* p_data);

            static Common::Status UnpackTensor(const LotusIR::TensorProto& p_tensor, /*out*/ std::vector<float>* p_data);

            static Common::Status UnpackTensor(const LotusIR::TensorProto& p_tensor, /*out*/ std::vector<int32_t>* p_data);

            static Common::Status UnpackTensor(const LotusIR::TensorProto& p_tensor, /*out*/ std::vector<bool>* p_data);

            static Common::Status UnpackTensor(const LotusIR::TensorProto& p_tensor, /*out*/ std::vector<int64_t>* p_data);

        private:

            static bool IsLittleEndianOrder();

            template <typename T>
            static void UnpackTensorWithRawData(const LotusIR::TensorProto& p_tensor, /*out*/ std::vector<T>* p_data)
            {
                auto& raw_data = p_tensor.raw_data();
                auto buff = raw_data.c_str();
                size_t typeSize = sizeof(T);

                for (size_t i = 0; i < raw_data.size(); i += typeSize)
                {
                    buff += i;

                    T result;
                    if (IsLittleEndianOrder())
                    {
                        memcpy((void*)&result, (void*)buff, typeSize);
                    }
                    else
                    {
                        const char* tempBytes = reinterpret_cast<char*>(&result);
                        for (size_t j = 0; j < typeSize; ++j)
                        {
                            memcpy((void*)&tempBytes[j], (void*)&buff[typeSize - 1 - i], sizeof(char));
                        }
                    }
                    p_data->push_back(result);
                }
            }
        };
    }
}



#endif