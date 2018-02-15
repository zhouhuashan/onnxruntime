// @@COPYRIGHT@@
#ifndef _LOTUS_CORE_PLATFORM_WINDOWS_TRACE_LOGGING_MANAGER_H_
#define _LOTUS_CORE_PLATFORM_WINDOWS_TRACE_LOGGING_MANAGER_H_

#include "core/platform/windows/trace_logging_supported.h"

#ifdef LOTUS_ETW_TRACE_LOGGING_SUPPORTED

#include <atomic>

#include <Windows.h>
#include <TraceLoggingProvider.h>

namespace Lotus {
    // unit test fixture
    namespace test { class TraceLoggingManagerTest; }

    /**
    * Manages registration and unregistration of the trace logging provider and
    * provides safe access to it through access handles. In particular, for a
    * given provider, calls to TraceLogging APIs should not be made during
    * registration or unregistration. This can be ensured by calling the APIs
    * while having a valid access handle.
    */
    class TraceLoggingManager {
        friend class test::TraceLoggingManagerTest;

    public:
        using ProviderHandle = TraceLoggingHProvider;

        /**
        * The access handle ensures safe access to the trace logging provider.
        */
        class AccessHandle {
            friend class TraceLoggingManager;

        public:
            ~AccessHandle();

            AccessHandle(AccessHandle&& other);
            AccessHandle& operator=(AccessHandle&& other);


            AccessHandle(const AccessHandle&) = delete;
            AccessHandle& operator=(const AccessHandle&) = delete;

            /**
            * Gets the trace logging provider. If the handle is not valid, returns null.
            */
            ProviderHandle GetProvider() const;

            /**
            * Gets whether the handle is valid.
            */
            bool IsValid() const { return valid_; }

        private:
            AccessHandle(TraceLoggingManager& manager);

        private:
            TraceLoggingManager* manager_;
            bool valid_;
        };

    public:
        TraceLoggingManager(const TraceLoggingManager&) = delete;
        TraceLoggingManager& operator=(TraceLoggingManager&) = delete;

        /**
        * Gets an access handle. If it is valid, then TraceLogging APIs may be
        * used with the trace logging provider while the handle is in scope.
        */
        AccessHandle Access();

        static TraceLoggingManager& Instance();

    private:
        TraceLoggingManager();
        ~TraceLoggingManager();

    private:
        std::atomic<bool> enabled_;
        std::atomic<int32_t> use_count_;
    };
}

#endif // LOTUS_ETW_TRACE_LOGGING_SUPPORTED

#endif // _LOTUS_CORE_PLATFORM_WINDOWS_TRACE_LOGGING_MANAGER_H_
