#include "core/platform/windows/trace_logging_manager.h"

#ifdef LOTUS_ETW_TRACE_LOGGING_SUPPORTED

#include <cassert>
#include <thread>

namespace Lotus {
namespace {
TRACELOGGING_DEFINE_PROVIDER(g_trace_logging_provider_handle, "LotusRtTraceLoggingProvider",
                             // {3111CE8F-F10A-48EB-8EF2-43720FC4CC90}
                             (0x3111ce8f, 0xf10a, 0x48eb, 0x8e, 0xf2, 0x43, 0x72, 0xf, 0xc4, 0xcc, 0x90));
}

// TraceLoggingManager::AccessHandle functions

TraceLoggingManager::AccessHandle::AccessHandle(TraceLoggingManager& manager)
    : manager_{&manager}, valid_{manager_->enabled_} {
  if (valid_) ++manager_->use_count_;
}

TraceLoggingManager::AccessHandle::~AccessHandle() {
  if (valid_) --manager_->use_count_;
}

TraceLoggingManager::AccessHandle::AccessHandle(AccessHandle&& other)
    : manager_{other.manager_},
      valid_{std::exchange(other.valid_, false)} {
}

TraceLoggingManager::AccessHandle&
TraceLoggingManager::AccessHandle::operator=(TraceLoggingManager::AccessHandle&& other) {
  if (&other != this) {
    if (valid_) --manager_->use_count_;
    manager_ = other.manager_;
    valid_ = std::exchange(other.valid_, false);
  }
  return *this;
}

TraceLoggingHProvider TraceLoggingManager::AccessHandle::GetProvider() const {
  return valid_ ? g_trace_logging_provider_handle : nullptr;
}

// TraceLoggingManager functions

TraceLoggingManager::TraceLoggingManager()
    : enabled_{false}, use_count_{0} {
  enabled_ =
      SUCCEEDED(::TraceLoggingRegister(g_trace_logging_provider_handle));
}

TraceLoggingManager::~TraceLoggingManager() {
  if (!enabled_) return;

  assert(use_count_ >= 0);

  enabled_ = false;
  // from this point, we assume that use_count_ will only decrease

  // give any outstanding usages some time to finish
  const auto time_limit = std::chrono::steady_clock::now() + std::chrono::milliseconds{15};
  while (use_count_ > 0 &&
         std::chrono::steady_clock::now() < time_limit) {
    std::this_thread::yield();
  }

  assert(use_count_ >= 0);

  // clean up
  if (use_count_ == 0) {
    ::TraceLoggingUnregister(g_trace_logging_provider_handle);
  }

  // TODO what happens if we time out?
}

TraceLoggingManager::AccessHandle TraceLoggingManager::Access() {
  return AccessHandle(*this);
}

TraceLoggingManager& TraceLoggingManager::Instance() {
  static TraceLoggingManager instance{};
  return instance;
}
}  // namespace Lotus

#endif  // LOTUS_ETW_TRACE_LOGGING_SUPPORTED