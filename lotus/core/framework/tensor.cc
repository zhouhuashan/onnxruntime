#include "core/framework/tensor.h"
#include "core/framework/allocatormgr.h"

namespace Lotus
{
    TensorShape::TensorShape() : TensorShape(std::vector<int64_t>())
    {
    }

    TensorShape::TensorShape(const std::vector<int64_t>& dims) : m_dims(dims)
    {
    }

    TensorShape::TensorShape(const TensorShape& p_other)
    {
        m_dims.assign(p_other.m_dims.begin(), p_other.m_dims.end());
    }

    int64_t& TensorShape::operator[](int p_idx)
    {
        //Since we don't have status in return value, if p_idex invalid, let stl throw exception.
        return m_dims[p_idx];
    }

    int64_t TensorShape::Size() const
    {
        int64_t size = 1;
        for (int i = 0; i < m_dims.size(); i++)
        {
            if (m_dims[i] < 0)
                throw std::logic_error("Can't calculate size for a un-resolved tensor shape");
            size *= m_dims[i];
        }
        //should we cache the size? as multiple operation may be expensive.
        return size;
    }

    TensorShape TensorShape::Slice(int p_dimstart, int p_dimend) const
    {
        if (p_dimstart < 0 || p_dimstart > p_dimend || p_dimend > m_dims.size())
            throw std::logic_error("Invliad tensor shape slice argument.");

        return TensorShape(std::vector<int64_t>(m_dims.begin() + p_dimstart, m_dims.begin() + p_dimend));
    }

    Tensor::Tensor() : Tensor(DataTypeImpl::GetType<float>())
    {
    }

    Tensor::Tensor(MlDataType p_type) : Tensor(p_type, TensorShape(std::vector<int64_t>(1, 0)), nullptr, AllocatorManager::Instance()->GetArena(CPU).Info())
    {
    }
    
    Tensor::Tensor(MlDataType p_type,
        const TensorShape& p_shape, 
        void* p_data, 
        AllocatorInfo& alloc, 
        const int64_t offset)
        : m_alloc_info(alloc)
    {
        init(p_type, p_shape, p_data, alloc, offset);
    }

    void Tensor::init(MlDataType p_type, const TensorShape& p_shape, void* p_data, AllocatorInfo& alloc, const int64_t bytes_offset)
    {
        m_dtype = p_type;
        m_shape = p_shape;
        m_pData = p_data;
        m_alloc_info = alloc;
        m_byte_offset = bytes_offset;
    }

    Tensor::Tensor(const Tensor& src)
        : m_dtype(src.m_dtype)
        , m_alloc_info(src.m_alloc_info)
        , m_shape(src.m_shape)
        , m_pData(src.m_pData)
        , m_byte_offset(src.m_byte_offset)
    {
    }


}
