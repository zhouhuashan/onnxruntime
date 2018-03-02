#ifndef CORE_FRAMEWORK_OP_KERNEL_H
#define CORE_FRAMEWORK_OP_KERNEL_H

#include "core/common/exceptions.h"
#include "core/common/status.h"
#include "core/framework/execution_frame.h"
#include "core/framework/ml_value.h"
#include "core/framework/tensor.h"
#include "core/graph/graph.h"

namespace Lotus {

  class OpKernel;
  
  class OpKernelInfo
  {
  public:
    explicit OpKernelInfo(const LotusIR::Node& node)
      : m_node(node) {}

    ~OpKernelInfo();

    template<typename T>
    Status GetAttr(const std::string& name, T* value) const;

    template<typename T>
    Status GetAttr(const std::string& name, std::vector<T>& values) const;
            
    const LotusIR::Node& OpDef() const { return m_node; }

    AllocatorInfo* GetAllocatorInfo();

  private:
    const LotusIR::Node& m_node;
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
    T* Output(int index) const;
      
  private:
    ExecutionFrame* m_execution_frame = nullptr;
    OpKernel* m_kernel = nullptr;
    ExecutionFrame::NodeArgValue* m_arg_start = nullptr;
  };

  class OpKernel {
  public:
    typedef std::function<void()> DoneCallback;

    explicit OpKernel(OpKernelInfo* info)
      : m_node(info->OpDef()),
        m_alloc(info->GetAllocatorInfo()) {
      m_input_start_index.resize(m_node.InputArgCount().size());
      size_t index = 0;
      for (size_t i = 0; i < m_input_start_index.size(); i++) {
        m_input_start_index[i] = index;
        index += m_node.InputArgCount()[i];
      }
    }

    // The total number of inputs.
    size_t num_inputs() const
    {
      return m_node.InputDefs().size();
    }

    // The total number of outputs.    
    size_t num_outputs() const
    {
      return m_node.OutputDefs().size();
    }
    
    // Starting index for the i-th input argument.
    size_t input_start_index(int arg_index) const
    {
      return m_input_start_index[arg_index];
    }

    // The number of inputs for the i-th input argument.
    size_t input_size(int arg_index) const
    {
      return m_node.InputArgCount()[arg_index];
    }
    
    virtual void Compute(OpKernelContext* context) = 0;
    virtual void ComputeAsync(OpKernelContext* context, DoneCallback done) = 0;

    const AllocatorInfo& allocator() { return *m_alloc; }

  private:
    AllocatorInfo* m_alloc;
    const LotusIR::Node& m_node;
    std::vector<size_t> m_input_start_index;
  };
}
#endif  // CORE_FRAMEWORK_OP_KERNEL_H
