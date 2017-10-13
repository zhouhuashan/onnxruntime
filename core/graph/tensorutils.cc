#include "core/graph/tensorutils.h"

namespace Lotus
{
    namespace Utils
    {
        bool TensorUtils::IsLittleEndianOrder()
        {
            int n = 1;
            return (*(char*)&n == 1);
        }

        void TensorUtils::UnpackTensor(const LotusIR::TensorProto& p_tensor, /*out*/ std::vector<std::string>* p_data)
        {
            if (LotusIR::TensorProto_DataType_STRING != p_tensor.data_type())
            {
                return;
            }

            for (auto& elem : p_tensor.string_data())
            {
                p_data->push_back(elem);
            }
        }

        void TensorUtils::UnpackTensor(const LotusIR::TensorProto& p_tensor, /*out*/ std::vector<float>* p_data)
        {
            if (LotusIR::TensorProto_DataType_FLOAT != p_tensor.data_type())
            {
                return;
            }

            if (!p_tensor.has_raw_data())
            {
                UnpackTensorWithRawData(p_tensor, p_data);
                return;
            }

            for (auto elem : p_tensor.float_data())
            {
                p_data->push_back(elem);
            }
        }

        void TensorUtils::UnpackTensor(const LotusIR::TensorProto& p_tensor, /*out*/ std::vector<int32_t>* p_data)
        {
            if (LotusIR::TensorProto_DataType_INT32 != p_tensor.data_type())
            {
                return;
            }

            if (!p_tensor.has_raw_data())
            {
                UnpackTensorWithRawData(p_tensor, p_data);
                return;
            }

            for (auto elem : p_tensor.int32_data())
            {
                p_data->push_back(elem);
            }
        }

        void TensorUtils::UnpackTensor(const LotusIR::TensorProto& p_tensor, /*out*/ std::vector<bool>* p_data)
        {
            if (LotusIR::TensorProto_DataType_INT32 != p_tensor.data_type())
            {
                return;
            }

            if (!p_tensor.has_raw_data())
            {
                UnpackTensorWithRawData(p_tensor, p_data);
                return;
            }

            for (auto elem : p_tensor.int32_data())
            {
                p_data->push_back(static_cast<bool>(elem));
            }
        }
    }
}