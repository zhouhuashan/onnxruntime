#include "core/framework/execution_frame.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"

namespace Lotus {

ExecutionFrame::ExecutionFrame(const std::unordered_map<std::string, MLValue>& feeds,
                               const std::vector<std::string>& output_names,
                               const SessionState& session_state)
    : session_state_(session_state) {
  Init(session_state.GetGraph(), feeds, output_names);
  InitArenas();
}

Status ExecutionFrame::AllocateTensorWithSelfOwnBuffer(const int index,
                                                       const MLDataType element_type,
                                                       const AllocatorInfo& location,
                                                       const TensorShape& shape) {
  LOTUS_ENFORCE(index >= 0 && index < node_values_.size());
  auto value = node_values_[index];
  LOTUS_ENFORCE(!value->IsAllocated());
  IAllocator* alloc = GetArena(location);
  void* buffer = alloc->Alloc(element_type->Size() * shape.Size());
  Tensor* tensor = new Tensor(
      element_type, 
      shape, 
      std::move(BufferUniquePtr(buffer, BufferDeleter(alloc))), 
      location);
  value->Init(tensor, 
              DataTypeImpl::GetType<Tensor>(),
              DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
  return Status::OK();
}

Status ExecutionFrame::AllocateTensorWithPreAllocateBuffer(const int offset,
                                                           void* pBuffer,
                                                           const MLDataType element_type,
                                                           const AllocatorInfo& location,
                                                           const TensorShape& shape)
{
  LOTUS_ENFORCE(offset >= 0 && offset < node_values_.size());
  auto value = node_values_[offset];
  LOTUS_ENFORCE(!value->IsAllocated());

  Tensor* tensor = new Tensor(element_type, 
                              shape, 
                              pBuffer, 
                              location);

  value->Init(tensor,
              DataTypeImpl::GetType<Tensor>(),
              DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
  return Status::OK();
}

void ExecutionFrame::Release(const int offset)
{
  LOTUS_ENFORCE(offset >= 0 && offset < node_offsets_.size());
  node_values_[offset]->Reset();
}

void ExecutionFrame::Init(const LotusIR::Graph* graph,
                          const std::unordered_map<string, MLValue>& feeds,
                          const std::vector<string>& output_names)
{
  LOTUS_ENFORCE(graph);

  //1. resize the node_offsets and all_value_ vector        
  auto num_nodes = graph->NumberOfNodes();
  node_offsets_.resize(num_nodes);
        
  all_values_.resize(session_state_.GetMaxMLValueIdx() + 1);
  
  //3. handle feed in values
  for (auto it = feeds.begin(); it != feeds.end(); it++)
  {
    int index;
    Common::Status status = session_state_.GetMLValueIdx(it->first, &index);
    LOTUS_ENFORCE(status.IsOK());
    // we are sharing the underline tensor/object for MLValue
    all_values_[index] = it->second;
  }
        
  //4. Todo: handle the weights.

  //5. set node args

  // TODO const_cast is needed due to the lack of a const iterator in the graph
  Graph* p_graph = const_cast<Graph*>(graph);
  
  for (auto node_it = p_graph->Nodes_begin(); node_it != p_graph->Nodes_end(); ++node_it) {
    auto node = *node_it;
    LOTUS_ENFORCE(node && node->Index() < node_offsets_.size());
    node_offsets_[node->Index()] = static_cast<int>(node_values_.size());
    auto& inputs = node->InputDefs();
    for (auto def : inputs)
    {
      SetupNodeArg(def);
    }
    auto& outputs = node->OutputDefs();
    for (auto def : outputs)
    {
      SetupNodeArg(def);
    }
  }
        
  //6. for outputs, we may limit the buffer strategy, for example,
  // output tensor should always use its own buffer. TBD
  UNUSED_PARAMETER(output_names);
}

void ExecutionFrame::SetupNodeArg(LotusIR::NodeArg* arg)
{
  LOTUS_ENFORCE(arg);
  auto& name = arg->Name();
  int index;
  Common::Status status = session_state_.GetMLValueIdx(name, &index);
  LOTUS_ENFORCE(status.IsOK());
  node_values_.push_back(&all_values_[index]);
}
}
