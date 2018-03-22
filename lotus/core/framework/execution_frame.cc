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

Status ExecutionFrame::AllocateMLValueTensorSelfOwnBuffer(int mlvalue_index,
                                                          const MLDataType element_type,
                                                          const AllocatorInfo& location,
                                                          const TensorShape& shape) {
  LOTUS_ENFORCE(mlvalue_index >= 0 && mlvalue_index < all_values_.size());
  MLValue* p_mlvalue = &all_values_[mlvalue_index];
  return AllocateMLValueTensorSelfOwnBuffer(p_mlvalue, element_type, location, shape);
}

Status ExecutionFrame::AllocateMLValueTensorSelfOwnBuffer(MLValue* p_mlvalue,
                                                          const MLDataType element_type,
                                                          const AllocatorInfo& location,
                                                          const TensorShape& shape) {
  if (p_mlvalue->IsAllocated()) {
    return Status::OK();
  }

  IAllocator* alloc = GetArena(location);
  void* buffer = alloc->Alloc(element_type->Size() * shape.Size());
  Tensor* tensor = new Tensor(element_type,
                              shape,
                              std::move(BufferUniquePtr(buffer, BufferDeleter(alloc))),
                              location);
  p_mlvalue->Init(tensor,
                  DataTypeImpl::GetType<Tensor>(),
                  DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
  return Status::OK();
}

Status ExecutionFrame::AllocateTensorWithSelfOwnBuffer(const int index,
                                                       const MLDataType element_type,
                                                       const AllocatorInfo& location,
                                                       const TensorShape& shape) {
  LOTUS_ENFORCE(index >= 0 && index < node_values_.size());
  auto value = &all_values_[node_values_[index]];
  return AllocateMLValueTensorSelfOwnBuffer(value, element_type, location, shape);
}

Status ExecutionFrame::AllocateMLValueTensorPreAllocateBuffer(int mlvalue_index_to_allocate,
                                                              int mlvalue_index_reuse,
                                                              const MLDataType element_type,
                                                              const AllocatorInfo& location,
                                                              const TensorShape& shape) {
  LOTUS_ENFORCE(mlvalue_index_to_allocate >= 0 && mlvalue_index_to_allocate < all_values_.size());
  MLValue* p_mlvalue = &all_values_[mlvalue_index_to_allocate];

  LOTUS_ENFORCE(mlvalue_index_reuse >= 0 && mlvalue_index_reuse < all_values_.size());
  MLValue* p_mlvalue_reuse = &all_values_[mlvalue_index_reuse];

  Tensor* reuse_tensor = p_mlvalue_reuse->GetMutable<Tensor>();
  void* reuse_buffer = reuse_tensor->GetRaw();

  return AllocateTensorWithPreAllocateBufferHelper(p_mlvalue, reuse_buffer, element_type, location, shape);
}

Status ExecutionFrame::AllocateTensorWithPreAllocateBufferHelper(MLValue* p_mlvalue,
                                                                 void* pBuffer,
                                                                 const MLDataType element_type,
                                                                 const AllocatorInfo& location,
                                                                 const TensorShape& shape) {
  if (p_mlvalue->IsAllocated()) {
    return Common::Status::OK();
  }
  Tensor* tensor = new Tensor(element_type,
                              shape,
                              pBuffer,
                              location);
  p_mlvalue->Init(tensor,
                  DataTypeImpl::GetType<Tensor>(),
                  DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());

  return Common::Status::OK();
}

Status ExecutionFrame::AllocateTensorWithPreAllocateBuffer(const int offset,
                                                           void* pBuffer,
                                                           const MLDataType element_type,
                                                           const AllocatorInfo& location,
                                                           const TensorShape& shape) {
  LOTUS_ENFORCE(offset >= 0 && offset < node_values_.size());
  auto value = &all_values_[node_values_[offset]];
  return AllocateTensorWithPreAllocateBufferHelper(value, pBuffer, element_type, location, shape);
}

void ExecutionFrame::Release(const int offset) {
  LOTUS_ENFORCE(offset >= 0 && offset < node_offsets_.size());
  all_values_[node_values_[offset]].Reset();
}

// This method is not thread safe!
Tensor* ExecutionFrame::get_or_create_tensor(int index, const TensorShape& shape) {
  LOTUS_ENFORCE(index >= 0 && index < node_values_.size());
  auto mlvalue_index = node_values_[index];
  LOTUS_ENFORCE(mlvalue_index >= 0 && mlvalue_index < all_values_.size());
  auto p_mlvalue = &all_values_[mlvalue_index];

  if (p_mlvalue->IsAllocated()) {
    // The tensor has already been allocated.
    // TODO: Check the size of the allocated tensor with given shape,
    // if they match each other, then return, else throw error.
    // TODO: type also needs to be checked and then use static_cast.
    Tensor* tensor = p_mlvalue->GetMutable<Tensor>();
    LOTUS_ENFORCE(tensor->shape() == shape);
    return tensor;
  } else {
    // It's not allocated, then allocate it with given shape and return.
    // TODO: at this point, we should already know the location and dtype
    // for the tensor, the graph should be able to tell us. But now graph
    // don't have it. So here hack to default as CPU and float.

    // perform allocation based on the allocation plan
    Status s = AllocateAsPerAllocationPlan(mlvalue_index, shape);
    LOTUS_ENFORCE(s.IsOK());
    return p_mlvalue->GetMutable<Tensor>();
  }
}

Common::Status ExecutionFrame::AllocateAsPerAllocationPlan(int mlvalue_index,
                                                           const TensorShape& shape) {
  const SequentialExecutionPlan* p_seq_exec_plan = session_state_.GetExecutionPlan();
  const auto& alloc_plan = p_seq_exec_plan->allocation_plan;
  LOTUS_ENFORCE(mlvalue_index >= 0 && mlvalue_index < alloc_plan.size());
  const auto& per_alloc_plan = alloc_plan[mlvalue_index];

  // TODO: both alloc_info and ml_data_type will be supplied by the allocation
  // plan later. This is a hack for now.
  auto alloc_info = AllocatorManager::Instance()->GetArena(CPU).Info();
  auto ml_data_type = DataTypeImpl::GetType<float>();

  AllocKind alloc_kind = per_alloc_plan.alloc_kind;
  switch (alloc_kind) {
    case AllocKind::kAllocate: {
      LOTUS_RETURN_IF_ERROR(AllocateMLValueTensorSelfOwnBuffer(mlvalue_index,
                                                               ml_data_type,
                                                               alloc_info,
                                                               shape));
      break;
    }
    case AllocKind::kReuse: {
      int reuse_mlvalue_index = per_alloc_plan.reused_buffer;
      LOTUS_RETURN_IF_ERROR(AllocateMLValueTensorPreAllocateBuffer(mlvalue_index,
                                                                   reuse_mlvalue_index,
                                                                   ml_data_type,
                                                                   alloc_info,
                                                                   shape));
      break;
    }
    default:
      return Common::Status(Common::LOTUS, Common::FAIL, "Invalid allocation kind");
  }

  return Common::Status::OK();
}

void ExecutionFrame::Init(const LotusIR::Graph* graph,
                          const std::unordered_map<string, MLValue>& feeds,
                          const std::vector<string>& output_names) {
  LOTUS_ENFORCE(graph);

  //1. resize the node_offsets and all_value_ vector
  auto num_nodes = graph->NumberOfNodes();
  node_offsets_.resize(num_nodes);

  all_values_.resize(session_state_.GetMaxMLValueIdx() + 1);

  //2. handle feed in values
  for (auto it = feeds.begin(); it != feeds.end(); it++) {
    int index;
    Common::Status status = session_state_.GetMLValueIdx(it->first, &index);
    LOTUS_ENFORCE(status.IsOK());
    // we are sharing the underline tensor/object for MLValue
    all_values_[index] = it->second;
  }

  //3. TODO: handle the weights.

  //4. set node args

  // TODO const_cast is needed due to the lack of a const iterator in the graph
  Graph* p_graph = const_cast<Graph*>(graph);

  for (auto node_it = p_graph->Nodes_begin(); node_it != p_graph->Nodes_end(); ++node_it) {
    auto node = *node_it;
    LOTUS_ENFORCE(node && node->Index() < node_offsets_.size());
    node_offsets_[node->Index()] = static_cast<int>(node_values_.size());
    auto& inputs = node->InputDefs();
    for (auto def : inputs) {
      SetupNodeArg(def);
    }
    auto& outputs = node->OutputDefs();
    for (auto def : outputs) {
      SetupNodeArg(def);
    }
  }

  //5. for outputs, we may limit the buffer strategy, for example,
  // output tensor should always use its own buffer. TBD
  UNUSED_PARAMETER(output_names);
}

void ExecutionFrame::SetupNodeArg(LotusIR::NodeArg* arg) {
  LOTUS_ENFORCE(arg);
  auto& name = arg->Name();
  int index;
  Common::Status status = session_state_.GetMLValueIdx(name, &index);
  LOTUS_ENFORCE(status.IsOK());
  node_values_.push_back(index);
}
}  // namespace Lotus
