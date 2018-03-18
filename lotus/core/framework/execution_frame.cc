#include "core/framework/execution_frame.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"

namespace Lotus {

  ExecutionFrame::ExecutionFrame(const std::unordered_map<std::string, MLValue>& feeds,
                                 const std::vector<std::string>& output_names,
                                 const SessionState& session_state) {
    Init(session_state.GetGraph(), feeds, output_names, session_state);
    InitArenas();
  }

  Status ExecutionFrame::AllocateTensorWithSelfOwnBuffer(
            const int index, const MLDataType element_type,
            const AllocatorInfo& location, const TensorShape& shape) {
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

  Status ExecutionFrame::AllocateTensorWithPreAllocateBuffer(
            const int offset, void* pBuffer, const MLDataType element_type,
            const AllocatorInfo& location, const TensorShape& shape) {
    LOTUS_ENFORCE(offset >= 0 && offset < node_values_.size());
    auto value = node_values_[offset];
    LOTUS_ENFORCE(!value->IsAllocated());

    Tensor* tensor = new Tensor(element_type, shape, pBuffer, location);
    value->Init(tensor,
                DataTypeImpl::GetType<Tensor>(),
                DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
    return Status::OK();
  }

  void ExecutionFrame::Release(const int offset) {
    LOTUS_ENFORCE(offset >= 0 && offset < node_offsets_.size());
    node_values_[offset]->Reset();
  }

  void ExecutionFrame::Init(const LotusIR::Graph* graph,
                            const std::unordered_map<string, MLValue>& feeds,
                            const std::vector<string>& output_names,
                            const SessionState& session_state) {
    LOTUS_ENFORCE(graph);
    // 1. construct the value name to index map
    // It seems not efficient to construct this map everytime
    // If planner could provide this, we can pass in the map.
    // TODO: avoid const_cast here since this operation can be performed
    // in the inference session once and avoided each time execution is
    // called
    std::vector<LotusIR::NODEINDEX>* nodes;
    auto status = const_cast<LotusIR::Graph*>(graph)->GetNodesInTopologicalOrder(&nodes);
    LOTUS_ENFORCE(status.IsOK());
    auto num_nodes = nodes->size();
    node_offsets_.resize(num_nodes);
    int current = 0;
    for (int i = 0; i < num_nodes; i++) {
      auto node = graph->GetNode((*nodes)[i]);
      auto& inputs = node->InputDefs();
      for (auto def : inputs) {
        if (value_name_to_index_.find(def->Name()) ==
            value_name_to_index_.end())
          value_name_to_index_[def->Name()] = current++;
      }
      auto& outputs = node->OutputDefs();
      for (auto def : outputs) {
        if (value_name_to_index_.find(def->Name()) ==
            value_name_to_index_.end())
          value_name_to_index_[def->Name()] = current++;
      }
    }

    // 2. resize the all_value_ vector
    auto num_values = value_name_to_index_.size();
    all_values_.resize(num_values);

    // 3. handle feed in values
    for (auto it = feeds.begin(); it != feeds.end(); it++) {
      auto index_it = value_name_to_index_.find(it->first);
      LOTUS_ENFORCE(index_it != value_name_to_index_.end());
      // we are sharing the underline tensor/object for MLValue
      all_values_[index_it->second] = it->second;
    }

    // 4. Todo: handle the weights.
    UNUSED_PARAMETER(session_state);

    // 5. set node args
    for (int i = 0; i < num_nodes; i++) {
      auto node = graph->GetNode((*nodes)[i]);
      LOTUS_ENFORCE(node && node->Index() < node_offsets_.size());
      node_offsets_[node->Index()] = (int)node_values_.size();
      auto& inputs = node->InputDefs();
      for (auto def : inputs) {
        SetupNodeArg(def, value_name_to_index_);
      }
      auto& outputs = node->OutputDefs();
      for (auto def : outputs) {
        SetupNodeArg(def, value_name_to_index_);
      }
    }

    // 6. for outputs, we may limit the buffer strategy, for example,
    // output tensor should always use its own buffer. TBD
    UNUSED_PARAMETER(output_names);
  }
}
