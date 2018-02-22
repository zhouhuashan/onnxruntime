#ifndef CORE_FRAMEWORK_OP_KERNEL_H
#define CORE_FRAMEWORK_OP_KERNEL_H

#include "core/common/status.h"
#include "core/framework/tensor.h"
#include "core/common/exceptions.h"
#include "core/graph/graph.h"
#include "core/framework/ml_value.h"

using namespace Lotus::Common;


namespace Lotus {
    class OpKernelInfo
    {
    public:
        explicit OpKernelInfo(const LotusIR::Node& node) :
            operator_def_(node) {}

        ~OpKernelInfo();

        template<typename T>
        Status GetAttr(const std::string& name, T* value) const;

        template<typename T>
        Status GetAttr(const std::string& name, std::vector<T>& values) const;
            
        const LotusIR::Node& OpDef() const { return operator_def_; }

    private:
        const LotusIR::Node& operator_def_;

    };

    class OpKernelContext {
    public:
        typedef std::unordered_map<std::string, size_t> ArgMap;
        /*
        *input/output names: the name of each inputs such X, Y, Input0, etc
        *input/output values: the values of all input, one input can have multiple values
        *inoput/output count: the number of values that each input has.
        * E.g. input_name={X, Y}, input_value={tensor0, tensor1, tensor2, tensor3, tensor4}, input_count={4,1}
        * It means: X={tensor0, tensor1, tensor2, tensor3}, Y={tensor4}
        * This is the same idea used in LotusIR::Node
        */
        OpKernelContext(std::vector<std::string> input_names, std::vector<std::string> output_names,
            std::vector<MLValue> input_values, std::vector<MLValue> output_values,
            std::vector<size_t> input_count, std::vector<size_t> output_count) :
            input_values_(input_values),
            output_values_(output_values){

            //waiting for logging impl 
            //CHECK_EQ(input_names.size(), input_count.size()); 
            //CHECK_EQ(output_names.size(), output_count.size());
            if (input_names.size() > 0) {
                input_table_[input_names[0]] = 0;
            }
            for (size_t i = 1; i < input_count.size(); ++i) {
                //we assume every input/output has an unique name
                input_table_[input_names[i]] = input_count[i-1];
            }

            if (output_names.size() > 0) {
                output_table_[output_names[0]] = 0;
            }
            for (size_t i = 0; i < output_count.size(); ++i) {
                //we assume every input/output has an unique name
                output_table_[output_names[i]] = output_count[i];
            }
        }

        ~OpKernelContext() {};

        //starting index in input_values
        size_t Input_Index(const std::string& name) const {
            return FindArgIndex(name, input_table_);
        }

        //starting index in output_values
        size_t Output_Index(const std::string& name) const {
            return FindArgIndex(name, output_table_);
        }

        template<typename T>
        const T* Input(int index) const {
            MLValue value = input_values_[index].m_pData;
            if (Is_Type<T>(value)) {
                return reinterpret_cast<Tensor*>(value);
            }
            else {
                throw TypeMismatchException();
            }
        }

        template<typename T>
        T* Output(int index) const {
            MLValue value = output_values_[index].m_pData;
            if (Is_Type<T>(value)) {
                return reinterpret_cast<Tensor*>(value);
            }
            else {
                throw TypeMismatchException();
            }
        }
      
    private:
        ArgMap input_table_; // input_name--starting_index mapping
        ArgMap output_table_; // output_name--starting_index mapping
        std::vector<MLValue> input_values_;
        std::vector<MLValue> output_values_;
        std::vector<size_t> arg_count_;

        inline size_t FindArgIndex(const std::string& name, const ArgMap& table) const{
            auto it = table.find(name);
            if (it != table.end()) {
                return it->second;
            }
            else {
                return 0;
            }
        }

        //Notes:
        //1. once we support more types besides tensor, this function has to change.
        //2. it's better to provide a type checking api in data_types.h
       
        template<typename T> 
        bool Is_Type(MLValue value) {
            if (typeid(T) == typeid(Tensor)) {
                if (value.m_type == DataTypeImpl::Tensor_FLOAT
                || value.m_type == DataTypeImpl::Tensor_UINT8
                || value.m_type == DataTypeImpl::Tensor_INT8
                || value.m_type == DataTypeImpl::Tensor_UINT16
                || value.m_type == DataTypeImpl::Tensor_INT16
                || value.m_type == DataTypeImpl::Tensor_INT32
                || value.m_type == DataTypeImpl::Tensor_INT64
                || value.m_type == DataTypeImpl::Tensor_STRING
                || value.m_type == DataTypeImpl::Tensor_BOOL
                || value.m_type == DataTypeImpl::Tensor_FLOAT16
                || value.m_type == DataTypeImpl::Tensor_DOUBLE
                || value.m_type == DataTypeImpl::Tensor_UINT32
                || value.m_type == DataTypeImpl::Tensor_UINT64
                || value.m_type == DataTypeImpl::Tensor_COMPLEX64
                || value.m_type == DataTypeImpl::Tensor_COMPLEX128)
                    return true;
            }
            return false;
        }
    };


    class OpKernel {
    public:
        typedef std::function<void()> DoneCallback;

        explicit OpKernel(OpKernelInfo* info, AllocatorInfo alloc) 
            :info_(info), alloc_(alloc) {}

        virtual void Compute(OpKernelContext* context) = 0;
        virtual void ComputeAsync(OpKernelContext* context, DoneCallback done) = 0;

        const AllocatorInfo allocator() { return alloc_; }
        

    private:
        OpKernelInfo* info_;
        AllocatorInfo alloc_;
    };

    // Map from operator name to kernels. 
    typedef OpKernel* (*KernelCreateFn)(OpKernelInfo*);
    typedef std::unordered_multimap<std::string, KernelCreateFn> KernelRegistry;
}
#endif  // CORE_FRAMEWORK_OP_KERNEL_H