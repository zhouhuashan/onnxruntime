// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/utils.h"

#include "core/graph/graph_viewer.h"

#include "core/framework/execution_frame.h"
#include "core/framework/execution_providers.h"
#include "core/framework/kernel_def_builder.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/parallel_executor.h"
#include "core/framework/session_state.h"
#include "core/framework/sequential_executor.h"

namespace onnxruntime {
namespace utils {

const KernelDef* GetKernelDef(const KernelRegistryManager& kernel_registry,
                              const onnxruntime::Node& node) {
  const KernelCreateInfo* kernel_create_info = nullptr;
  const KernelDef* kernel_def = nullptr;

  if (kernel_registry.SearchKernelRegistry(node, &kernel_create_info).IsOK()) {
    kernel_def = kernel_create_info->kernel_def.get();
  }

  return kernel_def;
}

AllocatorPtr GetAllocator(const ExecutionProviders& exec_providers, const OrtAllocatorInfo& allocator_info) {
  auto exec_provider = exec_providers.Get(allocator_info);
  if (exec_provider == nullptr) {
    return nullptr;
  }

  return exec_provider->GetAllocator(allocator_info.id, allocator_info.mem_type);
}

AllocatorPtr GetAllocator(const SessionState& session_state, const OrtAllocatorInfo& allocator_info) {
  return GetAllocator(session_state.GetExecutionProviders(), allocator_info);
}

common::Status AllocateHelper(const IExecutionProvider& execution_provider,
                              int device_id,
                              const Tensor& fetched_tensor,
                              MLValue& output_mlvalue) {
  auto allocator = execution_provider.GetAllocator(device_id, OrtMemTypeDefault);
  if (!allocator) {
    return Status(common::ONNXRUNTIME, common::FAIL, "invalid allocator");
  }

  void* buffer = nullptr;
  if (fetched_tensor.Size() != 0) {
    buffer = allocator->Alloc(fetched_tensor.Size());
    if (!buffer) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to allocate buffer. Execution provider type=",
                             execution_provider.Type());
    }
  }

  std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(fetched_tensor.DataType(),
                                                              fetched_tensor.Shape(),
                                                              buffer,
                                                              allocator->Info(),
                                                              allocator);
  output_mlvalue.Init(p_tensor.release(),
                      DataTypeImpl::GetType<Tensor>(),
                      DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());

  return Status::OK();
}

const std::string& GetNodeInputProviderType(const SessionState::NodeInfo& info) {
  // the input index will be std::numeric_limits<size_t>::max() if it's an implicit input to a control flow node.
  // the input will be processed fully when executing the subgraph that consumes the implicit input.
  bool implicit_input = info.index == std::numeric_limits<size_t>::max();

  // node may declare input_mem_type to be on CPU explicitly
  // skip implicit inputs as they don't have a valid 'index' value
  bool node_input_on_cpu = !implicit_input &&
                           info.kci && MemTypeOnCpuExplicitly(info.kci->kernel_def->InputMemoryType(info.index));

  // need a std::string that doesn't go away for kCpuExecutionProvider so we can return a reference.
  static const std::string cpu_execution_provider{onnxruntime::kCpuExecutionProvider};

  auto& required_provider_type = node_input_on_cpu ? cpu_execution_provider
                                                   : info.p_node->GetExecutionProviderType();

  return required_provider_type;
}

//static common::Status MapGraphInputsToMLValueIdxs(const InputDefList& graph_inputs_including_initializers,
//                                                  const MLValueNameIdxMap& mlvalue_name_idx_map,
//                                                  std::vector<FeedsFetchesOrder::EntryInfo>& graph_inputs_to_mlvalue_idxs) {
//  auto status = Status::OK();
//
//  graph_inputs_to_mlvalue_idxs.reserve(graph_inputs_including_initializers.size());
//
//  for (const auto& graph_input : graph_inputs_including_initializers) {
//    int idx;
//    status = mlvalue_name_idx_map.GetIdx(graph_input->Name(), idx);
//    ORT_RETURN_IF_ERROR(status);
//
//    graph_inputs_to_mlvalue_idxs.push_back({graph_input->Name(), idx});
//  }
//
//  return status;
//}

//void VectorizeFeeds(const NameMLValMap& feeds, const InputDefList& graph_inputs_including_initializers,
//                    std::vector<const MLValue*>& vectorized_feeds) {
//  vectorized_feeds.resize(graph_inputs_including_initializers.size(), nullptr);
//
//  int idx = 0;
//  auto feeds_end = feeds.cend();
//  for (const auto& input : graph_inputs_including_initializers) {
//    auto input_in_feeds = feeds.find(input->Name());
//    if (input_in_feeds != feeds_end) {
//      vectorized_feeds[idx] = &input_in_feeds->second;
//    }
//
//    ++idx;
//  }
//}

//common::Status MapOutputsToMLValueIdxs(const std::vector<std::string>& output_names,
//                                       const MLValueNameIdxMap& mlvalue_name_idx_map,
//                                       std::vector<FeedsFetchesOrder::EntryInfo>& outputs_to_mlvalue_idxs) {
//  auto status = Status::OK();
//
//  outputs_to_mlvalue_idxs.reserve(output_names.size());
//
//  for (const auto& output : output_names) {
//    int idx;
//    status = mlvalue_name_idx_map.GetIdx(output, idx);
//    ORT_RETURN_IF_ERROR(status);
//
//    outputs_to_mlvalue_idxs.push_back({output, idx});
//  }
//
//  return status;
//}

static common::Status MapNamesToMLValueIdxs(const std::vector<std::string>& names,
                                            const MLValueNameIdxMap& mlvalue_name_idx_map,
                                            std::vector<int>& mlvalue_idxs) {
  auto status = Status::OK();

  mlvalue_idxs.reserve(names.size());

  for (const auto& name : names) {
    int idx;
    status = mlvalue_name_idx_map.GetIdx(name, idx);
    ORT_RETURN_IF_ERROR(status);

    mlvalue_idxs.push_back(idx);
  }

  return status;
}

Status FeedsFetchesOrder::Create(const std::vector<std::string> feed_names,
                                 const std::vector<std::string>& fetch_names,
                                 const MLValueNameIdxMap& mlvalue_name_idx_map,
                                 std::unique_ptr<FeedsFetchesOrder>& feed_fetch_order) {
  feed_fetch_order = std::make_unique<FeedsFetchesOrder>();

  auto status = MapNamesToMLValueIdxs(feed_names,
                                      mlvalue_name_idx_map,
                                      feed_fetch_order->feeds_mlvalue_idxs_);
  ORT_RETURN_IF_ERROR(status);

  status = MapNamesToMLValueIdxs(fetch_names, mlvalue_name_idx_map, feed_fetch_order->fetches_mlvalue_idxs_);

  return status;
}

static Status NoCopy(const MLValue& orig_value, MLValue& new_value) {
  new_value = orig_value;
  return Status::OK();
}

// TODO should we handle the case of one input name feeding 2 nodes placed on different devices?
common::Status CopyOneInputAcrossDevices(const SessionState& session_state,
                                         const std::string& input_name,
                                         const MLValue& orig_mlvalue,
                                         MLValue& new_mlvalue,
                                         bool& needed_copy,
                                         std::vector<std::function<Status(const MLValue&, MLValue&)>>* copiers) {
  needed_copy = false;

  //TODO: make it configurable
  const int target_device_id = 0;
  std::vector<SessionState::NodeInfo> node_info_vec;
  ORT_RETURN_IF_ERROR(session_state.GetInputNodeInfo(input_name, node_info_vec));

  auto& exec_providers = session_state.GetExecutionProviders();

  do {
    // currently we only support one device per input. see SessionState::AddInputNameToNodeInfoMapping for more
    // info on the logic to create the node_info_vec.
    // for (auto& node_info : node_info_vec) {
    auto& node_info = node_info_vec.front();

    if (node_info.p_node == nullptr) {
      // dummy entry for an input that we didn't find a use of in the graph.
      // use the input as is given we don't believe it's actually needed.
      new_mlvalue = orig_mlvalue;
      break;
    }

    if (!orig_mlvalue.IsTensor()) {
      // copying not supported for non-tensor types
      new_mlvalue = orig_mlvalue;
      break;
    }

    auto& required_provider_type = GetNodeInputProviderType(node_info);
    auto& input_tensor = orig_mlvalue.Get<Tensor>();
    auto& input_tensor_loc = input_tensor.Location();

    auto* p_input_provider = exec_providers.Get(input_tensor_loc);
    if (!p_input_provider) {
      p_input_provider = exec_providers.Get(onnxruntime::kCpuExecutionProvider);
      ORT_ENFORCE(p_input_provider);
    }

    //no copy for TRT
    if (required_provider_type == onnxruntime::kTRTExecutionProvider) {
      new_mlvalue = orig_mlvalue;
      break;
    }

    auto input_provider_type = p_input_provider->Type();
    if (input_provider_type == required_provider_type && input_tensor_loc.mem_type == OrtMemTypeDefault) {
      new_mlvalue = orig_mlvalue;
      break;
    }

    // If a node requires input on cpu and input tensor is allocated with pinned memory allocator, don't do copy
    if (required_provider_type == onnxruntime::kCpuExecutionProvider &&
        (input_tensor_loc.mem_type == OrtMemTypeCPU ||
         input_tensor_loc.mem_type == OrtMemTypeCPUOutput)) {
      new_mlvalue = orig_mlvalue;
      break;
    }

    auto* required_provider = exec_providers.Get(required_provider_type);
    ORT_ENFORCE(required_provider);
    auto copier = [&required_provider_type,
                   &required_provider,
                   &p_input_provider,
                   &target_device_id](const MLValue& feed_value, MLValue& new_value) {
      const auto& feed_tensor = feed_value.Get<Tensor>();
      ORT_RETURN_IF_ERROR(utils::AllocateHelper(*required_provider, target_device_id, feed_tensor, new_value));
      auto* new_tensor = new_value.GetMutable<Tensor>();

      if (required_provider_type != onnxruntime::kCpuExecutionProvider) {
        ORT_RETURN_IF_ERROR(required_provider->CopyTensor(feed_tensor, *new_tensor));
      } else {
        ORT_RETURN_IF_ERROR(p_input_provider->CopyTensor(feed_tensor, *new_tensor));
      }
    };

    // ORT_RETURN_IF_ERROR(utils::AllocateHelper(*required_provider, target_device_id, input_tensor, new_mlvalue));
    // auto* new_tensor = new_mlvalue.GetMutable<Tensor>();

    // our CPU exec provider doesn't support copy from GPU->CPU
    //if (required_provider_type != onnxruntime::kCpuExecutionProvider) {
    //  ORT_RETURN_IF_ERROR(required_provider->CopyTensor(input_tensor, *new_tensor));
    //} else {
    //  ORT_RETURN_IF_ERROR(p_input_provider->CopyTensor(input_tensor, *new_tensor));
    //}

    ORT_RETURN_IF_ERROR(copier(orig_mlvalue, new_mlvalue));

    if (copiers)
      copiers->push_back(std::move(copier));

    needed_copy = true;

    // } loop of node_info_vec
  } while (false);

  if (!needed_copy && copiers)
    copiers->push_back(NoCopy);

  return Status::OK();
}

// copies inputs across devices only if required
static common::Status CopyInputsAcrossDevices(const SessionState& session_state,
                                              //const NameMLValMap& orig_feeds,
                                              // NameMLValMap& new_feeds,
                                              const std::vector<std::string>& feed_names,
                                              std::vector<MLValue> orig_feeds,
                                              std::vector<MLValue> new_feeds,
                                              bool& needed_copy,
                                              std::vector<std::function<Status(const MLValue&, MLValue&)>>& copiers) {
  bool copied = false;
  size_t num_feeds = orig_feeds.size();
  ORT_ENFORCE(feed_names.size() == num_feeds);

  new_feeds.resize(num_feeds);

  if (!copiers.empty()) {
    ORT_ENFORCE(num_feeds == copiers.size());
    needed_copy = true;

    // use cached info
    for (size_t idx = 0; idx < num_feeds; ++idx) {
      ORT_RETURN_IF_ERROR(copiers[idx](orig_feeds[idx], new_feeds[idx]));
    }

  } else {
    for (size_t idx = 0; idx < num_feeds; ++idx) {
      bool copied_this_input = false;
      ORT_RETURN_IF_ERROR(CopyOneInputAcrossDevices(session_state, feed_names[idx], orig_feeds[idx], new_feeds[idx],
                                                    copied_this_input, &copiers));
      copied = copied || copied_this_input;
    }

    needed_copy = copied;
  }

  return Status::OK();
}

static std::pair<bool, size_t> Contains(const std::vector<std::string>& output_names,
                                        const std::string& name) {
  auto it = std::find(std::begin(output_names), std::end(output_names), name);
  if (it == output_names.end()) {
    return {false, 0};
  }
  return {true, it - output_names.begin()};
}

// ensures pre-allocated outputs match the node providers.
static common::Status MatchOutputsWithProviders(const SessionState& session_state,
                                                const std::vector<std::string>& output_names,
                                                std::vector<MLValue>& fetches,
                                                std::vector<MLValue>& new_fetches) {
  const auto& execution_providers = session_state.GetExecutionProviders();

  if (fetches.empty()) {
    fetches.resize(output_names.size());
  }

  new_fetches.resize(output_names.size());

  std::set<std::string> seen_outputs;
  auto p_graph = session_state.GetGraphViewer();
  ORT_ENFORCE(p_graph);

  std::vector<const Node*> leaf_nodes;

  for (auto& node : p_graph->Nodes()) {
    if (node.OutputNodesBegin() == node.OutputNodesEnd()) {
      // This is a leaf node (without any output node).
      leaf_nodes.push_back(&node);
    }
  }

  std::pair<bool, size_t> found;
  //  for (auto& node : p_graph->Nodes()) {  // TODO optimize this
  for (auto* p_node : leaf_nodes) {
    auto& node = *p_node;
    if (seen_outputs.size() == fetches.size()) {
      break;
    }

    for (auto* arg : node.OutputDefs()) {
      if (!arg->Exists() ||
          !(found = Contains(output_names, arg->Name())).first) {
        continue;
      }

      seen_outputs.insert(arg->Name());
      size_t idx = found.second;
      MLValue orig_mlvalue = fetches[idx];

      if (orig_mlvalue.IsAllocated()) {
        if (!orig_mlvalue.IsTensor()) {
          new_fetches[idx] = fetches[idx];
          continue;
        }

        auto& node_provider_type = node.GetExecutionProviderType();
        auto& orig_tensor = orig_mlvalue.Get<Tensor>();
        auto& orig_tensor_loc = orig_tensor.Location();
        auto* tensor_provider = execution_providers.Get(orig_tensor_loc);
        if (!tensor_provider) {
          tensor_provider = execution_providers.Get(onnxruntime::kCpuExecutionProvider);
        }

        auto tensor_provider_type = tensor_provider->Type();
        if (node_provider_type == tensor_provider_type) {
          new_fetches[idx] = fetches[idx];
          continue;
        }

        // leave the new_fetches[idx] as it is since it'll get allocated on the appropriate
        // provider by the op kernel context when requested.
        continue;

      } else {
        // TODO this seems unnecessary. if fetches[idx] wasn't allocated, is there any point copying it to new_fetches[idx]?
        new_fetches[idx] = fetches[idx];
        continue;
      }
    }
  }

  return Status::OK();
}

// copies outputs across devices only if required
static common::Status CopyOutputsAcrossDevices(const SessionState& session_state,
                                               std::vector<MLValue>& fetches,
                                               std::vector<MLValue>& user_fetches,
                                               bool& needed_copy,
                                               std::vector<bool>& outputs_needing_copy) {
  needed_copy = false;
  auto& execution_providers = session_state.GetExecutionProviders();

  for (size_t idx = 0, end = fetches.size(); idx < end; ++idx) {
    auto& fetched_mlvalue = fetches[idx];
    if (!fetched_mlvalue.IsTensor()) {
      user_fetches[idx] = fetched_mlvalue;
      continue;
    }

    auto& fetched_tensor = fetched_mlvalue.Get<Tensor>();
    auto& fetched_tensor_location = fetched_tensor.Location();
    auto* p_fetched_provider = execution_providers.Get(fetched_tensor_location);
    if (!p_fetched_provider) {
      p_fetched_provider = execution_providers.Get(onnxruntime::kCpuExecutionProvider);
      ORT_ENFORCE(p_fetched_provider);
    }

    auto fetched_provider_type = p_fetched_provider->Type();
    auto& output_mlvalue = user_fetches[idx];

    if (!output_mlvalue.IsAllocated()) {
      if (fetched_provider_type != onnxruntime::kCpuExecutionProvider) {
        ORT_RETURN_IF_ERROR(utils::AllocateHelper(*execution_providers.Get(onnxruntime::kCpuExecutionProvider),
                                                  0,
                                                  fetched_tensor,
                                                  output_mlvalue));
      } else {
        user_fetches[idx] = fetched_mlvalue;
        continue;
      }
    }

    Tensor* p_output_tensor = output_mlvalue.GetMutable<Tensor>();
    auto& output_tensor_loc = p_output_tensor->Location();
    auto* p_output_provider = execution_providers.Get(output_tensor_loc);
    if (!p_output_provider) {
      p_output_provider = execution_providers.Get(onnxruntime::kCpuExecutionProvider);
      ORT_ENFORCE(p_output_provider);
    }

    auto output_provider_type = p_output_provider->Type();

    if (output_provider_type == fetched_provider_type || fetched_tensor_location.mem_type == OrtMemTypeCPUOutput) {
      user_fetches[idx] = fetched_mlvalue;
      continue;
    }

    needed_copy = true;

    // our CPU exec provider doesn't support copy from GPU->CPU
    if (fetched_provider_type != onnxruntime::kCpuExecutionProvider) {
      ORT_RETURN_IF_ERROR(p_fetched_provider->CopyTensor(fetched_tensor, *p_output_tensor));
    } else {
      ORT_RETURN_IF_ERROR(p_output_provider->CopyTensor(fetched_tensor, *p_output_tensor));
    }
  }

  return Status::OK();
}

common::Status ExecuteGraph(const SessionState& session_state,
                            const NameMLValMap& feeds,
                            const std::vector<std::string>& output_names,
                            std::vector<MLValue>& fetches,
                            const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                            bool sequential_execution,
                            const bool& terminate_flag,
                            const logging::Logger& logger,
                            DeviceCopyChecks& device_copy_checks) {
}

common::Status ExecuteGraph(const SessionState& session_state,
                            const std::vector<MLValue>& feeds,
                            const std::vector<int>& output_mlvalue_idx,
                            std::vector<MLValue>& fetches,
                            const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                            bool sequential_execution,
                            const bool& terminate_flag,
                            const logging::Logger& logger,
                            DeviceCopyChecks& device_copy_checks) {
  std::unique_ptr<IExecutor> p_exec;

  if (sequential_execution) {
    p_exec = std::unique_ptr<IExecutor>(new SequentialExecutor(terminate_flag));
  } else {
    p_exec = std::unique_ptr<IExecutor>(new ParallelExecutor(session_state, terminate_flag));
  }

  // If we know we don't need to check both inputs and outputs for copies, we can just execute.
  // If we only have one provider it's the CPU provider as that is always automatically registered, and if that's the
  // case we can also assume no copy to/from other devices is required.
  // TODO: When the different execution providers can share a single CPU Allocator we should be able to easily handle
  // checking if all execution providers are CPU based and skip the copy in that case
  if ((device_copy_checks.check_input_copy_needed == DeviceCopyCheck::Skip &&
       device_copy_checks.check_output_copy_needed == DeviceCopyCheck::Skip) /*||
      session_state.GetExecutionProviders().NumProviders() == 1*/
  ) {
    // no device copies are needed so simple execute
    ORT_RETURN_IF_ERROR(p_exec->Execute(session_state, feeds, output_names, fetches, fetch_allocators, logger));
  } else {
    bool copy_needed = false;
    const NameMLValMap* p_feeds = &feeds;
    std::vector<MLValue>* p_fetches = &fetches;

    NameMLValMap device_feeds;
    std::vector<MLValue> device_fetches;

    if (device_copy_checks.check_input_copy_needed == DeviceCopyCheck::Check) {
      ORT_RETURN_IF_ERROR(utils::CopyInputsAcrossDevices(session_state, feeds, device_feeds,
                                                         copy_needed, device_copy_checks.input_copy_needed));

      if (copy_needed) {
        p_feeds = &device_feeds;
      } else {
        device_copy_checks.check_input_copy_needed = DeviceCopyCheck::Skip;
      }
    }

    // if we are skipping copies of outputs, we don't need to match outputs with providers
    if (device_copy_checks.check_output_copy_needed == DeviceCopyCheck::Check) {
      ORT_RETURN_IF_ERROR(utils::MatchOutputsWithProviders(session_state, output_names, fetches, device_fetches));
      p_fetches = &device_fetches;
    }

    ORT_RETURN_IF_ERROR(p_exec->Execute(session_state, *p_feeds, output_names, *p_fetches, fetch_allocators,
                                        logger));

    if (device_copy_checks.check_output_copy_needed == DeviceCopyCheck::Check) {
      ORT_RETURN_IF_ERROR(utils::CopyOutputsAcrossDevices(session_state, device_fetches, fetches,
                                                          copy_needed, device_copy_checks.output_copy_needed));
      if (!copy_needed) {
        device_copy_checks.check_output_copy_needed = DeviceCopyCheck::Skip;
      }
    }
  }
  return Status::OK();
}

}  // namespace utils
}  // namespace onnxruntime
