#ifndef CORE_FRAMEWORK_OP_KERNEL_H
#define CORE_FRAMEWORK_OP_KERNEL_H

#include "core/framework/execution_frame.h"
#include "core/framework/ml_value.h"
#include "core/framework/tensor.h"
#include "core/framework/exceptions.h"
#include "core/graph/graph.h"
#include "core/graph/status.h"

using namespace Lotus::Common;

namespace Lotus {

  class OpKernel;
  
  class OpKernelInfo
  {
  public:
    explicit OpKernelInfo(const LotusIR::Node& node) : m_opdef(node) {}

    ~OpKernelInfo();

    template<typename T>
    Status GetAttr(const std::string& name, T* value) const;

    template<typename T>
    Status GetAttr(const std::string& name, std::vector<T>& values) const;
            
    const LotusIR::Node& OpDef() const { return m_opdef; }

    AllocatorInfo* GetAllocatorInfo();

  private:
    const LotusIR::Node& m_opdef;
  };

  class OpKernelContext {
  public:
    typedef std::unordered_map<std::string, size_t> ArgMap;
    
    explicit OpKernelContext(const LotusIR::Node& node, OpKernel* kernel,
                             ExecutionFrame* frame)
      : m_kernel(kernel),
        m_execution_frame(frame)
    {
      m_arg_start = frame->get_first_arg(node);
    }
    
    ~OpKernelContext() {};

    template<typename T>
    const T* Input(int index) const {
      ExecutionFrame::NodeArgValue value = m_arg_start[index];
      return reinterpret_cast<T*>(value->pData);
    }

    template<typename T>
    T* Output(int index) const {
      int num_inputs = m_kernel->num_inputs();
      ExecutionFrame::NodeArgValue value = m_arg_start[num_inputs + index];
      return reinterpret_cast<T*>(value->pData);
    }
      
  private:
    ExecutionFrame* m_execution_frame = nullptr;
    OpKernel* m_kernel = nullptr;
    ExecutionFrame::NodeArgValue* m_arg_start = nullptr;
  };

  class OpKernel {
  public:
    typedef std::function<void()> DoneCallback;

    explicit OpKernel(OpKernelInfo* info)
      : m_alloc(info->GetAllocatorInfo()) {}

    // The total number of inputs and outputs.
    int num_inputs() const { return 1; }
    int num_outputs() const { return 1; }
    
    // starting index in input_values
    size_t Input_Index(int arg_index) const;

    // starting index in output_values
    size_t Output_Index(int arg_index) const;

    // The number of inputs for the i-th input argument.
    int input_size(int arg_index) const;
    
    virtual void Compute(OpKernelContext* context) = 0;
    virtual void ComputeAsync(OpKernelContext* context, DoneCallback done) = 0;

    const AllocatorInfo& allocator() { return *m_alloc; }

  private:
    AllocatorInfo* m_alloc;
  };
}
#endif  // CORE_FRAMEWORK_OP_KERNEL_H
