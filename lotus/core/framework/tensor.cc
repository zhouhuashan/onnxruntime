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

    const int64_t& TensorShape::operator[](int p_idx) const
    {
        //Since we don't have status in return value, if p_idex invalid, let stl throw exception.
        return m_dims[p_idx];
    }

    int64_t TensorShape::Size() const
    {
        int64_t size = 1;
        for (int i = 0; i < m_dims.size(); i++)
        {
            LOTUS_ENFORCE(m_dims[i] >= 0, "Can't calculate size for a un-resolved tensor shape");
            size *= m_dims[i];
        }
        //should we cache the size? as multiple operation may be expensive.
        return size;
    }

    TensorShape TensorShape::Slice(int p_dimstart, int p_dimend) const
    {
        LOTUS_ENFORCE(p_dimstart >= 0 && p_dimstart <= p_dimend && p_dimend <= m_dims.size(), "Invliad tensor shape slice argument.");
        return TensorShape(std::vector<int64_t>(m_dims.begin() + p_dimstart, m_dims.begin() + p_dimend));
    }

    Tensor::Tensor() : 
        alloc_info_(AllocatorManager::Instance()->GetArena(CPU).Info()),
        p_unique_data_(BufferUniquePtr(nullptr, BufferDeleter()))
    {
        Init(DataTypeImpl::GetType<float>(),
            TensorShape(std::vector<int64_t>(1, 0)),
            UNKNOWN,
            nullptr,
            AllocatorManager::Instance()->GetArena(CPU).Info(),
            0);
    }

    Tensor::Tensor(MLDataType p_type) :
        alloc_info_(AllocatorManager::Instance()->GetArena(CPU).Info()),
        p_unique_data_(BufferUniquePtr(nullptr, BufferDeleter()))
    {
        Init(p_type,
            TensorShape(std::vector<int64_t>(1, 0)),
            UNKNOWN,
            nullptr,
            AllocatorManager::Instance()->GetArena(CPU).Info(),
            0);
    }
    
    Tensor::Tensor(MLDataType p_type,
        const TensorShape& p_shape, 
        BufferNakedPtr p_data,
        const AllocatorInfo& alloc, 
        const int64_t offset)
        : alloc_info_(alloc),
        p_unique_data_(BufferUniquePtr(nullptr, BufferDeleter()))
    {
        Init(p_type,
            p_shape,
            PREALLOCATEDBUFFER,
            p_data,
            alloc,
            offset);
    }

    Tensor::Tensor(MLDataType p_type,
        const TensorShape& p_shape,
        BufferUniquePtr p_data,
        const AllocatorInfo& alloc,
        const int64_t offset)
        : alloc_info_(alloc),
        p_unique_data_(std::move(p_data))
    {
        Init(p_type,
            p_shape,
            OWNEDBUFFER,
            nullptr,
            alloc,
            offset);
    }

    void Tensor::Init(MLDataType p_type, 
        const TensorShape& p_shape,
        BufferStrategy strategy,
        BufferNakedPtr p_raw_data,
        const AllocatorInfo& alloc,
        const int64_t offset)
    {
        dtype_ = p_type;
        shape_ = p_shape;
        buffer_strategy_ = strategy;
        p_naked_data_ = p_raw_data;
        alloc_info_ = alloc;
        byte_offset_ = offset;
    }

    Tensor::Tensor(Tensor&& other)
        : dtype_(other.dtype_),
        shape_(other.shape_),
        alloc_info_(other.alloc_info_),
        byte_offset_(other.byte_offset_),
        buffer_strategy_(other.buffer_strategy_)
    {
        if (other.buffer_strategy_ == OWNEDBUFFER)
        {
            p_unique_data_ = std::move(other.p_unique_data_);
            p_naked_data_ = nullptr;
        }
        else
        {
            p_naked_data_ = other.p_naked_data_;
            p_unique_data_ = nullptr;
        }

        other.dtype_ = DataTypeImpl::GetType<float>();
        other.shape_ = TensorShape(std::vector<int64_t>(1, 0));
        other.buffer_strategy_ = UNKNOWN;
        other.byte_offset_ = 0;
        other.p_unique_data_ = nullptr;
    }

    Tensor& Tensor::operator=(Tensor&& other)
    {
        if (this != &other)
        {
            dtype_ = other.dtype_;
            shape_ = other.shape_;
            alloc_info_ = other.alloc_info_;
            byte_offset_ = other.byte_offset_;
            buffer_strategy_ = other.buffer_strategy_;
            if (other.buffer_strategy_ == OWNEDBUFFER)
            {
                p_unique_data_ = std::move(other.p_unique_data_);
                p_naked_data_ = nullptr;
            }
            else
            {
                p_naked_data_ = other.p_naked_data_;
                p_unique_data_ = nullptr;
            }

            other.dtype_ = DataTypeImpl::GetType<float>();
            other.shape_ = TensorShape(std::vector<int64_t>(1, 0));
            other.buffer_strategy_ = UNKNOWN;
            other.byte_offset_ = 0;
            other.p_unique_data_ = nullptr;
        }
        return *this;
    }

    Tensor::Tensor(const Tensor& src)
        : dtype_(src.dtype_)
        , alloc_info_(src.alloc_info_)
        , shape_(src.shape_)
        , byte_offset_(src.byte_offset_)
    {
        // it may be better to refactor it a little bit to make it a compile error
        // but right now just keep it simple first.
        if (src.buffer_strategy_ == OWNEDBUFFER)
        {
            LOTUS_THROW("Can't copy tensor with its owned buffer. Please transfer ownership by move");
        }
        else if (src.buffer_strategy_ == PREALLOCATEDBUFFER)
        {
            buffer_strategy_ = PREALLOCATEDBUFFER;
            p_naked_data_ = src.p_naked_data_;
        }
        else
        {
            buffer_strategy_ = UNKNOWN;
            p_naked_data_ = nullptr;
            p_unique_data_ = nullptr;
        }
    }

    Tensor& Tensor::ShallowCopy(const Tensor& other)
    {
        // smiliar as above
        LOTUS_ENFORCE(other.buffer_strategy_ != OWNEDBUFFER);
        if (this != &other)
        {
            dtype_ = other.dtype_;
            alloc_info_ = other.alloc_info_;
            shape_ = other.shape_;
            byte_offset_ = other.byte_offset_;
            buffer_strategy_ = other.buffer_strategy_;
            p_naked_data_ = other.p_naked_data_;
            p_naked_data_ = nullptr;
        }
        return *this;
    }


}
