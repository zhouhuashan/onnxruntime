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

Status SetupFeedsFetchesInfo(std::function<void(std::vector<std::string>& feed_names)> feed_name_populator,
                             const std::vector<std::string>& output_names,
                             const MLValueNameIdxMap& mlvalue_name_idx_map,
                             FeedsFetchesInfo& info) {
  feed_name_populator(info.feed_names);
  info.output_names = output_names;

  auto status = MapNamesToMLValueIdxs(info.feed_names, mlvalue_name_idx_map, info.feeds_mlvalue_idxs);
  ORT_RETURN_IF_ERROR(status);

  status = MapNamesToMLValueIdxs(output_names, mlvalue_name_idx_map, info.fetches_mlvalue_idxs);
  return status;
}

Status FeedsFetchesManager::Create(const std::vector<std::string>& feed_names,
                                   const std::vector<std::string>& output_names,
                                   const MLValueNameIdxMap& mlvalue_name_idx_map,
                                   std::unique_ptr<FeedsFetchesManager>& feed_fetch_order) {
  FeedsFetchesInfo info;

  ORT_RETURN_IF_ERROR(SetupFeedsFetchesInfo(
      [&feed_names](std::vector<std::string>& feed_names_target) {
        // just copy feed_names into the std::vector from FeedsFetchesInfo
        feed_names_target = feed_names;
      },
      output_names, mlvalue_name_idx_map, info));

  // can't use std::make_unique to call a private ctor
  feed_fetch_order = std::unique_ptr<FeedsFetchesManager>(new FeedsFetchesManager(std::move(info)));

  return Status::OK();
}

Status FeedsFetchesManager::Create(const std::unordered_map<std::string, MLValue>& feeds,
                                   const std::vector<std::string>& output_names,
                                   const MLValueNameIdxMap& mlvalue_name_idx_map,
                                   std::unique_ptr<FeedsFetchesManager>& feed_fetch_order) {
  FeedsFetchesInfo info;
  ORT_RETURN_IF_ERROR(SetupFeedsFetchesInfo(
      [&feeds](std::vector<std::string>& feed_names_target) {
        feed_names_target.reserve(feeds.size());
        // copy the feed names from feeds ino the std::vector from FeedsFetchesInfo
        std::transform(feeds.cbegin(), feeds.cend(),
                       std::back_inserter(feed_names_target),
                       [](const std::pair<std::string, MLValue>& pair) { return pair.first; });
      },
      output_names, mlvalue_name_idx_map, info));

  // can't use std::make_unique to call a private ctor
  feed_fetch_order = std::unique_ptr<FeedsFetchesManager>(new FeedsFetchesManager(std::move(info)));

  return Status::OK();
}

static Status SimpleCopy(const MLValue& orig_value, MLValue& new_value) {
  new_value = orig_value;
  return Status::OK();
}

// TODO should we handle the case of one input name feeding 2 nodes placed on different devices?
common::Status CopyOneInputAcrossDevices(const SessionState& session_state,
                                         const std::string& input_name,
                                         const MLValue& orig_mlvalue,
                                         MLValue& new_mlvalue,
                                         bool& needed_copy,
                                         FeedsFetchesManager::MLValueCopyFunc* copier_out) {
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
                   required_provider,
                   p_input_provider,
                   target_device_id](const MLValue& feed_value, MLValue& new_value) {
      const auto& feed_tensor = feed_value.Get<Tensor>();
      ORT_RETURN_IF_ERROR(utils::AllocateHelper(*required_provider, target_device_id, feed_tensor, new_value));
      auto* new_tensor = new_value.GetMutable<Tensor>();

      if (required_provider_type != onnxruntime::kCpuExecutionProvider) {
        ORT_RETURN_IF_ERROR(required_provider->CopyTensor(feed_tensor, *new_tensor));
      } else {
        ORT_RETURN_IF_ERROR(p_input_provider->CopyTensor(feed_tensor, *new_tensor));
      }

      return Status::OK();
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

    if (copier_out)
      *copier_out = std::move(copier);

    needed_copy = true;

    // } loop of node_info_vec
  } while (false);

  if (!needed_copy && copier_out)
    *copier_out = SimpleCopy;

  return Status::OK();
}

common::Status CopyOneInputAcrossDevices(const SessionState& session_state,
                                         const std::string& input_name,
                                         const MLValue& orig_mlvalue,
                                         MLValue& new_mlvalue) {
  bool needed_copy;
  return CopyOneInputAcrossDevices(session_state, input_name, orig_mlvalue, new_mlvalue, needed_copy, nullptr);
}

// copies inputs across devices only if required
static common::Status CopyInputsAcrossDevices(const SessionState& session_state,
                                              //const NameMLValMap& orig_feeds,
                                              // NameMLValMap& new_feeds,
                                              const std::vector<std::string>& feed_names,
                                              std::vector<MLValue> orig_feeds,
                                              std::vector<MLValue> new_feeds,
                                              bool& needed_copy,
                                              std::vector<FeedsFetchesManager::MLValueCopyFunc>* copiers) {
  bool copied = false;
  size_t num_feeds = orig_feeds.size();
  ORT_ENFORCE(feed_names.size() == num_feeds);

  new_feeds.resize(num_feeds);

  // use cached copy logic if available
  if (copiers && !copiers->empty()) {
    ORT_ENFORCE(num_feeds == copiers->size());
    needed_copy = true;

    for (size_t idx = 0; idx < num_feeds; ++idx) {
      ORT_RETURN_IF_ERROR((*copiers)[idx](orig_feeds[idx], new_feeds[idx]));
    }

  } else {
    if (copiers) {
      copiers->reserve(num_feeds);
    }

    for (size_t idx = 0; idx < num_feeds; ++idx) {
      bool copied_this_input = false;
      std::function<Status(const MLValue&, MLValue&)> copier;
      ORT_RETURN_IF_ERROR(CopyOneInputAcrossDevices(session_state, feed_names[idx], orig_feeds[idx], new_feeds[idx],
                                                    copied_this_input, &copier));
      copied = copied || copied_this_input;

      if (copiers) {
        copiers->push_back(std::move(copier));
      }
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

// if pre-allocated outputs match the node providers, use them directly.
// if they don't match, the execution will populate the value in new_fetches, and we'll copy it to fetches
// later using CopyOutputsAcrossDevices
// if we setup new_fetches and it should be used in the Execute, use_new_fetches is set to true
static common::Status MatchOutputsWithProviders(const SessionState& session_state,
                                                const std::vector<std::string>& output_names,
                                                std::vector<MLValue>& fetches,
                                                std::vector<MLValue>& new_fetches,
                                                bool& use_new_fetches,
                                                std::vector<bool>* can_copy_to_new_fetches_cached_values) {
  ORT_ENFORCE(new_fetches.empty());

  use_new_fetches = false;

  // no allocated outputs, so nothing to look at here
  if (fetches.empty()) {
    return Status::OK();
  }

  const auto& execution_providers = session_state.GetExecutionProviders();
  auto num_outputs = output_names.size();

  if (can_copy_to_new_fetches_cached_values && !can_copy_to_new_fetches_cached_values->empty()) {
    // use the cached values
    ORT_ENFORCE(can_copy_to_new_fetches_cached_values->size() == num_outputs);

    // we assume the logic outside of here only calls with a populated use_fetch_when_executing_values if we do need
    // to copy to new_fetches
    use_new_fetches = true;

    auto& needs_copy = *can_copy_to_new_fetches_cached_values;
    for (size_t i = 0; i < num_outputs; ++i) {
      if (needs_copy[i]) {
        new_fetches[i] = fetches[i];
      }
    }

    return Status::OK();
  }

  // track which fetches can be copied to new_fetches and used directly in the execution.
  // this could turn out to be all values, in which case we don't need new_fetches.
  std::vector<bool> local_can_copy_flags(num_outputs, false);

  std::set<std::string> seen_outputs;
  auto p_graph = session_state.GetGraphViewer();
  ORT_ENFORCE(p_graph);

  std::pair<bool, size_t> found;
  for (auto& node : p_graph->Nodes()) {
    if (seen_outputs.size() == num_outputs) {
      break;
    }

    for (auto* arg : node.OutputDefs()) {
      if (!arg->Exists() ||
          !(found = Contains(output_names, arg->Name())).first) {
        continue;
      }

      seen_outputs.insert(arg->Name());
      size_t idx = found.second;
      const MLValue& orig_mlvalue = fetches[idx];

      if (orig_mlvalue.IsAllocated()) {
        if (!orig_mlvalue.IsTensor()) {
          local_can_copy_flags[idx] = true;
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
          local_can_copy_flags[idx] = true;
          continue;
        }

        // can't copy. a new value will be allocated during execution in new_fetches, and we will
        // copy that the orig_mlvalue in CopyOutputsAcrossDevices
        use_new_fetches = true;
        continue;
      } else {
        local_can_copy_flags[idx] = true;
      }
    }
  }

  // if we can copy all the values, we can just use fetches directly and don't need new_fetches.
  // if any are have false, we need to copy the ones we can to new_fetches.
  if (use_new_fetches) {
    new_fetches.resize(num_outputs);

    for (size_t idx = 0; idx < num_outputs; ++idx) {
      if (local_can_copy_flags[idx]) {
        new_fetches[idx] = fetches[idx];
      }
    }
  }

  if (can_copy_to_new_fetches_cached_values) {
    *can_copy_to_new_fetches_cached_values = local_can_copy_flags;
  }

  return Status::OK();
}

// copies outputs across devices only if required
static common::Status CopyOutputsAcrossDevices(const SessionState& session_state,
                                               std::vector<MLValue>& fetches,
                                               std::vector<MLValue>& user_fetches,
                                               bool& needed_copy,
                                               std::vector<FeedsFetchesManager::MLValueCopyFunc>* copiers) {
  needed_copy = false;
  auto num_outputs = fetches.size();

  // used the cached copy logic if available
  if (copiers && !copiers->empty()) {
    for (size_t idx = 0, end = num_outputs; idx < end; ++idx) {
      ORT_RETURN_IF_ERROR((*copiers)[idx](fetches[idx], user_fetches[idx]));
    }

    return Status::OK();
  }

  if (copiers) {
    copiers->reserve(num_outputs);
  }

  auto& execution_providers = session_state.GetExecutionProviders();

  // CPU execution provider is always registered so this is not null
  const auto* cpu_execution_provider = execution_providers.Get(onnxruntime::kCpuExecutionProvider);

  auto do_simple_copy = [&user_fetches, &copiers](MLValue& fetched, size_t idx) {
    user_fetches[idx] = fetched;
    if (copiers) {
      copiers->push_back(SimpleCopy);
    }
  };

  for (size_t idx = 0; idx < num_outputs; ++idx) {
    auto& fetched_mlvalue = fetches[idx];
    if (!fetched_mlvalue.IsTensor()) {
      do_simple_copy(fetched_mlvalue, idx);
      continue;
    }

    auto& fetched_tensor = fetched_mlvalue.Get<Tensor>();
    auto& fetched_tensor_location = fetched_tensor.Location();
    auto* p_fetched_provider = execution_providers.Get(fetched_tensor_location);
    if (!p_fetched_provider) {
      p_fetched_provider = cpu_execution_provider;
    }

    auto fetched_provider_type = p_fetched_provider->Type();
    auto& output_mlvalue = user_fetches[idx];

    if (!output_mlvalue.IsAllocated() && fetched_provider_type == onnxruntime::kCpuExecutionProvider) {
      do_simple_copy(fetched_mlvalue, idx);
      continue;
    }

    Tensor* p_output_tensor = output_mlvalue.GetMutable<Tensor>();
    auto& output_tensor_loc = p_output_tensor->Location();
    auto* p_output_provider = execution_providers.Get(output_tensor_loc);
    if (!p_output_provider) {
      p_output_provider = cpu_execution_provider;
    }

    auto output_provider_type = p_output_provider->Type();

    if (output_provider_type == fetched_provider_type || fetched_tensor_location.mem_type == OrtMemTypeCPUOutput) {
      do_simple_copy(fetched_mlvalue, idx);
      continue;
    }

    needed_copy = true;

    auto copy_between_providers = [&fetched_provider_type,
                                   p_fetched_provider,
                                   p_output_provider](const MLValue& fetched_mlvalue, MLValue& output_mlvalue) {
      auto& fetched_tensor = fetched_mlvalue.Get<Tensor>();

      if (!output_mlvalue.IsAllocated()) {
        ORT_RETURN_IF_ERROR(utils::AllocateHelper(*p_output_provider, 0, fetched_tensor, output_mlvalue));
      }

      Tensor* p_output_tensor = output_mlvalue.GetMutable<Tensor>();

      // our CPU exec provider doesn't support copy from GPU->CPU
      if (fetched_provider_type != onnxruntime::kCpuExecutionProvider) {
        ORT_RETURN_IF_ERROR(p_fetched_provider->CopyTensor(fetched_tensor, *p_output_tensor));
      } else {
        ORT_RETURN_IF_ERROR(p_output_provider->CopyTensor(fetched_tensor, *p_output_tensor));
      }

      return Status::OK();
    };

    ORT_RETURN_IF_ERROR(copy_between_providers(fetched_mlvalue, output_mlvalue));

    if (copiers) {
      copiers->push_back(std::move(copy_between_providers));
    }
  }

  return Status::OK();
}

common::Status ExecuteGraph(const SessionState& session_state,
                            FeedsFetchesManager& feeds_fetches_manager,
                            const std::vector<MLValue>& feeds,
                            std::vector<MLValue>& fetches,
                            const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                            bool sequential_execution,
                            const bool& terminate_flag,
                            const logging::Logger& logger,
                            bool cache_copy_info) {
  const auto& feeds_fetches_info = feeds_fetches_manager.GetFeedsFetchesInfo();

  std::unique_ptr<IExecutor> p_exec;

  if (sequential_execution) {
    p_exec = std::unique_ptr<IExecutor>(new SequentialExecutor(terminate_flag));
  } else {
    p_exec = std::unique_ptr<IExecutor>(new ParallelExecutor(session_state, terminate_flag));
  }

  auto device_copy_checks = feeds_fetches_manager.GetDeviceCopyChecks();
  // If we only have one provider it's the CPU provider as that is always automatically registered, and if that's the
  // case we can also assume no copy to/from other devices is required.
  // TODO: When the different execution providers can share a single CPU Allocator we should be able to easily handle
  // checking if all execution providers are CPU based and skip the copy in that case
  if (device_copy_checks.status == DeviceCopyCheck::NoCopy /*||
      session_state.GetExecutionProviders().NumProviders() == 1*/
                                                           // FIXME <--- TEMPORARY for testing purposes. Add dummy provider in unit test to force 'else' path to be tested
  ) {
    // no device copies are needed so simple execute
    ORT_RETURN_IF_ERROR(p_exec->Execute(session_state,
                                        feeds_fetches_info.feeds_mlvalue_idxs, feeds,
                                        feeds_fetches_info.fetches_mlvalue_idxs, fetches, fetch_allocators, logger));
  } else {
    // first execution we check and update. after that we use cached values
    bool check_all = device_copy_checks.status == DeviceCopyCheck::Check;
    bool copy_needed = false;

    const std::vector<MLValue>* p_feeds = &feeds;
    std::vector<MLValue>* p_fetches = &fetches;

    std::vector<MLValue> device_feeds;
    std::vector<MLValue> device_fetches;

    if (check_all || device_copy_checks.input_copy_needed == DeviceCopyCheck::Copy) {
      auto* copiers = cache_copy_info ? &feeds_fetches_manager.GetFeedsDeviceCopiers() : nullptr;
      ORT_RETURN_IF_ERROR(utils::CopyInputsAcrossDevices(session_state,
                                                         feeds_fetches_info.feed_names, feeds, device_feeds,
                                                         copy_needed, copiers));

      if (copy_needed) {
        p_feeds = &device_feeds;
      }

      if (check_all) {
        device_copy_checks.input_copy_needed = copy_needed ? DeviceCopyCheck::Copy : DeviceCopyCheck::NoCopy;
      }
    }

    if (check_all || device_copy_checks.copy_fetch_for_execution_needed == DeviceCopyCheck::Copy) {
      auto* can_copy_cache_info = cache_copy_info ? &feeds_fetches_manager.GetCanUseFetchDuringExecutionFlags() : nullptr;

      ORT_RETURN_IF_ERROR(utils::MatchOutputsWithProviders(session_state, feeds_fetches_info.output_names,
                                                           fetches, device_fetches,
                                                           copy_needed,  // did we copy to device_fetches
                                                           can_copy_cache_info));
      if (copy_needed) {
        p_fetches = &device_fetches;
      }

      if (check_all) {
        device_copy_checks.input_copy_needed = copy_needed ? DeviceCopyCheck::Copy : DeviceCopyCheck::NoCopy;
      }
    }

    ORT_RETURN_IF_ERROR(p_exec->Execute(session_state,
                                        feeds_fetches_info.feeds_mlvalue_idxs, *p_feeds,
                                        feeds_fetches_info.fetches_mlvalue_idxs, *p_fetches, fetch_allocators,
                                        logger));

    if (check_all || device_copy_checks.output_copy_needed == DeviceCopyCheck::Copy) {
      auto* copiers = cache_copy_info ? &feeds_fetches_manager.GetFetchesDeviceCopiers() : nullptr;

      ORT_RETURN_IF_ERROR(utils::CopyOutputsAcrossDevices(session_state, device_fetches, fetches,
                                                          copy_needed, copiers));
      if (check_all) {
        device_copy_checks.output_copy_needed = copy_needed ? DeviceCopyCheck::Copy : DeviceCopyCheck::NoCopy;
      }
    }

    // save the result of all the checks and use cached info next time
    if (check_all && cache_copy_info) {
      feeds_fetches_manager.SetDeviceCopyChecks(device_copy_checks);
    }
  }

  return Status::OK();
}

}  // namespace utils
}  // namespace onnxruntime
