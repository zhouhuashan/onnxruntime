// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/feeds_fetches_manager.h"

#include "core/framework/execution_providers.h"
#include "core/framework/mlvalue_name_idx_map.h"

namespace onnxruntime {
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

Status FeedsFetchesInfo::SetMLValueIdxs(const MLValueNameIdxMap& mlvalue_name_idx_map) {
  auto status = MapNamesToMLValueIdxs(feed_names, mlvalue_name_idx_map, feeds_mlvalue_idxs);
  ORT_RETURN_IF_ERROR(status);

  status = MapNamesToMLValueIdxs(output_names, mlvalue_name_idx_map, fetches_mlvalue_idxs);
  return status;
}

Status FeedsFetchesManager::Create(const std::vector<std::string>& feed_names,
                                   const std::vector<std::string>& output_names,
                                   const MLValueNameIdxMap& mlvalue_name_idx_map,
                                   std::unique_ptr<FeedsFetchesManager>& feed_fetch_manager) {
  FeedsFetchesInfo info;
  info.feed_names = feed_names;
  info.output_names = output_names;

  ORT_RETURN_IF_ERROR(info.SetMLValueIdxs(mlvalue_name_idx_map));

  feed_fetch_manager = std::make_unique<FeedsFetchesManager>(std::move(info));

  return Status::OK();
}

DeviceCopyCheck FeedsFetchesManager::CheckExecutionProviders(const ExecutionProviders& execution_providers) {
  bool all_cpu = true;
  for (const auto& execution_provider : execution_providers) {
    // TODO: Would be better if we can avoid calling GetAllocatorMap as it constructs a new vector every time.
    auto allocators = execution_provider->GetAllocatorMap();
    // this won't work as desired until multiple providers can share the CPU Allocator.
    // it will currently handle the scenario when only the CPUExecutionProvider is registered, and will work better
    // once the changes to share the CPU allocator are made.
    if (!std::all_of(allocators.cbegin(), allocators.cend(),
                     [](const AllocatorPtr& allocator) {
                       return strcmp(allocator->Info().name, CPU) == 0;
                     })) {
      all_cpu = false;
      break;
    }
  }

  if (all_cpu) {
    device_copy_checks_.status = DeviceCopyCheck::NoCopy;
  }

  return device_copy_checks_.status;
}

}  // namespace onnxruntime
