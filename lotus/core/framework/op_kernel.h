#ifndef CORE_FRAMEWORK_OP_KERNEL_H
#define CORE_FRAMEWORK_OP_KERNEL_H


#include "core/framework/execution_frame.h"
#include "core/framework/ml_value.h"
#include "core/common/status.h"
#include "core/framework/tensor.h"
#include "core/common/exceptions.h"
#include "core/graph/graph.h"

namespace Lotus {

  class OpKernelContext;
  
  class OpKernelInfo
  {
  public:
    explicit OpKernelInfo(const LotusIR::Node& node) : m_node(node) {}

    //Get a single attribute
    template<typename T>
    Status GetAttr(const std::string& name, T* value) const;

    //Get repeated attributes
    template<typename T>
    Status GetAttrs(const std::string& name, std::vector<T>& values) const;
            
    const LotusIR::Node& OpDef() const { return m_node; }

    AllocatorInfo* GetAllocatorInfo();

  private:
    const LotusIR::Node& m_node;
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
      virtual void ComputeAsync(OpKernelContext* context, DoneCallback done) {
          UNUSED_PARAMETER(context);
          UNUSED_PARAMETER(done);
          LOTUS_NOT_IMPLEMENTED;
      }

      const AllocatorInfo& allocator() { return *m_alloc; }

  private:
      AllocatorInfo* m_alloc;
      const LotusIR::Node& m_node;
      std::vector<size_t> m_input_start_index;
  };


  class OpKernelContext {
  public:
      typedef std::unordered_map<std::string, size_t> ArgMap;

      explicit OpKernelContext(const LotusIR::Node& node, OpKernel* kernel,
          ExecutionFrame* frame)
          : m_kernel(kernel),
          m_execution_frame(frame)
      {
          arg_start_index = frame->get_first_arg(node);
      }

      ~OpKernelContext() {};

      template<typename T>
      const T* Input(int index) const {
          return m_execution_frame->GetInput<T>(arg_start_index + index);
      }

      template<typename T>
      T* Output(int index) {
          auto output_arg_index = arg_start_index + static_cast<int>(m_kernel->num_inputs()) + index;
          return m_execution_frame->GetOutput<T>(output_arg_index);
      }

      // In the case that memory allocation has not been done for an output tensor,
      // The memory allocation will be done on-the-fly with given tensor shape.
      Tensor* Output(int index, const TensorShape& shape);

  private:
      ExecutionFrame* m_execution_frame = nullptr;
      OpKernel* m_kernel = nullptr;

      // The argument starting index in ExecutionFrame.
      int arg_start_index = -1;
  };

}
#endif  // CORE_FRAMEWORK_OP_KERNEL_H
