// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/basic_types.h"
#include "core/framework/allocator.h"
#include "core/framework/data_types.h"
#include "core/framework/framework_common.h"
#include "core/framework/iexecutor.h"
#include "core/framework/session_state.h"

namespace onnxruntime {
class ExecutionProviders;
class Graph;
class KernelDef;
class KernelRegistryManager;
class IExecutionProvider;
class MLValue;
class Node;
class Tensor;

namespace logging {
class Logger;
}

namespace utils {
const KernelDef* GetKernelDef(const KernelRegistryManager& kernel_registry,
                              const onnxruntime::Node& node);

AllocatorPtr GetAllocator(const ExecutionProviders& exec_providers, const OrtAllocatorInfo& allocator_info);

AllocatorPtr GetAllocator(const SessionState& session_state, const OrtAllocatorInfo& allocator_info);

common::Status AllocateHelper(const IExecutionProvider& execution_provider,
                              int device_id,
                              const Tensor& fetched_tensor,
                              MLValue& output_mlvalue);

const std::string& GetNodeInputProviderType(const SessionState::NodeInfo& info);

common::Status CopyOneInputAcrossDevices(const SessionState& session_state,
                                         const std::string& input_name,
                                         const MLValue& orig_mlvalue,
                                         MLValue& new_mlvalue,
                                         bool& needed_copy,
                                         std::vector<std::function<Status(const MLValue&, MLValue&)>>* copiers);

//common::Status CopyInputsAcrossDevices(const SessionState& session_state,
//                                       const NameMLValMap& orig_feeds,
//                                       NameMLValMap& new_feeds,
//                                       bool* needed_copy = nullptr);
//
//common::Status MatchOutputsWithProviders(const SessionState& session_state,
//                                         const std::vector<std::string>& output_names,
//                                         std::vector<MLValue>& fetches,
//                                         std::vector<MLValue>& new_fetches);
//
//common::Status CopyOutputsAcrossDevices(const SessionState& session_state,
//                                        std::vector<MLValue>& fetches,
//                                        std::vector<MLValue>& user_fetches,
//                                        bool* needed_copy = nullptr);

enum class DeviceCopyCheck {
  Skip,  // skip checking if a copy is needed
  Check  // check if a copy is needed
};

struct DeviceCopyChecks {
  DeviceCopyCheck check_input_copy_needed = DeviceCopyCheck::Check;
  DeviceCopyCheck check_output_copy_needed = DeviceCopyCheck::Check;
};

// convert an NameMLValMap to a vector of MLValue instances that match the order of GraphProto.inputs()
//void VectorizeFeeds(const NameMLValMap& feeds, const InputDefList& graph_inputs_including_initializers,
//                    std::vector<const MLValue>& vectorized_feeds);

class FeedsFetchesOrder {
 public:
  static Status Create(const std::vector<std::string> feed_names,
                       const std::vector<std::string>& fetch_names,
                       const MLValueNameIdxMap& mlvalue_name_idx_map,
                       std::unique_ptr<FeedsFetchesOrder>& feed_fetch_order);

  static Status Create(std::unordered_map<std::string, MLValue>& feeds,
                       const std::vector<std::string>& fetch_names,
                       const MLValueNameIdxMap& mlvalue_name_idx_map,
                       std::unique_ptr<FeedsFetchesOrder>& feed_fetch_order);

  //struct EntryInfo {
  //  EntryInfo(const std::string& name_in, int mlvalue_idx_in)
  //      : name{name_in}, mlvalue_idx{mlvalue_idx_in} {}
  //  const std::string name;
  //  const int mlvalue_idx;
  //};

  //const std::vector<EntryInfo>& GetFeedsInfo() const {
  //  return feeds_info_;
  //}

  //const std::vector<EntryInfo>& GetFetchesInfo() const {
  //  return fetches_info_;
  //}
  const std::vector<std::string>& GetFeedNames() const { return feed_names_; }
  const std::vector<std::string>& GetOutputNames() const { return output_names_; }

  const std::vector<int>& GetFeedsMLValueIdxs() const { return feeds_mlvalue_idxs_; }
  const std::vector<int>& GetFetchesMLValueIdxs() const { return fetches_mlvalue_idxs_; }

  std::vector<std::function<Status(const MLValue&, MLValue&)>>& GetFeedsDeviceCopiers() { return feeds_device_copiers_; }
  std::vector<std::function<Status(const MLValue&, MLValue&)>>& GetFetchesDeviceCopiers() { return fetches_device_copiers_; }

 private:
  FeedsFetchesOrder() = default;

  //std::vector<EntryInfo> feeds_info_;
  //std::vector<EntryInfo> fetches_info_;
  std::vector<std::string> feed_names_;
  std::vector<std::string> output_names_;

  std::vector<int> feeds_mlvalue_idxs_;
  std::vector<int> fetches_mlvalue_idxs_;

  std::vector<std::function<Status(const MLValue&, MLValue&)>> feeds_device_copiers_;
  std::vector<std::function<Status(const MLValue&, MLValue&)>> fetches_device_copiers_;
};

common::Status ExecuteGraph(const SessionState& session_state,
                            const NameMLValMap& feeds,
                            const std::vector<std::string>& output_names,
                            std::vector<MLValue>& fetches,
                            const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                            bool sequential_execution,
                            const bool& terminate_flag,
                            const logging::Logger& logger,
                            DeviceCopyChecks& device_copy_checks);

common::Status ExecuteGraph(const SessionState& session_state,
                            const std::vector<MLValue*>& feeds,
                            const std::vector<int>& output_mlvalue_idx,
                            std::vector<MLValue>& fetches,
                            const std::unordered_map<size_t, IExecutor::CustomAllocator>& fetch_allocators,
                            bool sequential_execution,
                            const bool& terminate_flag,
                            const logging::Logger& logger,
                            DeviceCopyChecks& device_copy_checks);

#define DispatchOnTensorType(tensor_type, function, ...)      \
  if (tensor_type == DataTypeImpl::GetType<float>())          \
    function<float>(__VA_ARGS__);                             \
  else if (tensor_type == DataTypeImpl::GetType<double>())    \
    function<double>(__VA_ARGS__);                            \
  else if (tensor_type == DataTypeImpl::GetType<int8_t>())    \
    function<int8_t>(__VA_ARGS__);                            \
  else if (tensor_type == DataTypeImpl::GetType<int16_t>())   \
    function<int16_t>(__VA_ARGS__);                           \
  else if (tensor_type == DataTypeImpl::GetType<int32_t>())   \
    function<int32_t>(__VA_ARGS__);                           \
  else if (tensor_type == DataTypeImpl::GetType<int64_t>())   \
    function<int64_t>(__VA_ARGS__);                           \
  else if (tensor_type == DataTypeImpl::GetType<uint8_t>())   \
    function<uint8_t>(__VA_ARGS__);                           \
  else if (tensor_type == DataTypeImpl::GetType<uint16_t>())  \
    function<uint16_t>(__VA_ARGS__);                          \
  else if (tensor_type == DataTypeImpl::GetType<uint32_t>())  \
    function<uint32_t>(__VA_ARGS__);                          \
  else if (tensor_type == DataTypeImpl::GetType<uint64_t>())  \
    function<uint64_t>(__VA_ARGS__);                          \
  else if (tensor_type == DataTypeImpl::GetType<bool>())      \
    function<bool>(__VA_ARGS__);                              \
  else if (tensor_type == DataTypeImpl::GetType<MLFloat16>()) \
    function<MLFloat16>(__VA_ARGS__);                         \
  else if (tensor_type == DataTypeImpl::GetType<BFloat16>())  \
  function<BFloat16>(__VA_ARGS__)

#define DispatchOnTensorTypeWithReturn(tensor_type, retval, function, ...) \
  if (tensor_type == DataTypeImpl::GetType<float>())                       \
    retval = function<float>(__VA_ARGS__);                                 \
  else if (tensor_type == DataTypeImpl::GetType<double>())                 \
    retval = function<double>(__VA_ARGS__);                                \
  else if (tensor_type == DataTypeImpl::GetType<int8_t>())                 \
    retval = function<int8_t>(__VA_ARGS__);                                \
  else if (tensor_type == DataTypeImpl::GetType<int16_t>())                \
    retval = function<int16_t>(__VA_ARGS__);                               \
  else if (tensor_type == DataTypeImpl::GetType<int32_t>())                \
    retval = function<int32_t>(__VA_ARGS__);                               \
  else if (tensor_type == DataTypeImpl::GetType<int64_t>())                \
    retval = function<int64_t>(__VA_ARGS__);                               \
  else if (tensor_type == DataTypeImpl::GetType<uint8_t>())                \
    retval = function<uint8_t>(__VA_ARGS__);                               \
  else if (tensor_type == DataTypeImpl::GetType<uint16_t>())               \
    retval = function<uint16_t>(__VA_ARGS__);                              \
  else if (tensor_type == DataTypeImpl::GetType<uint32_t>())               \
    retval = function<uint32_t>(__VA_ARGS__);                              \
  else if (tensor_type == DataTypeImpl::GetType<uint64_t>())               \
    retval = function<uint64_t>(__VA_ARGS__);                              \
  else if (tensor_type == DataTypeImpl::GetType<bool>())                   \
    retval = function<bool>(__VA_ARGS__);                                  \
  else if (tensor_type == DataTypeImpl::GetType<MLFloat16>())              \
    retval = function<MLFloat16>(__VA_ARGS__);                             \
  else if (tensor_type == DataTypeImpl::GetType<BFloat16>())               \
  retval = function<BFloat16>(__VA_ARGS__)

}  // namespace utils
}  // namespace onnxruntime
