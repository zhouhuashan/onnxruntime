#include "core/framework/init.h"
#include "core/framework/allocatormgr.h"

namespace Lotus {

    Status Initializer::EnsureInitialized(int* pargc, char*** pargv)
    {
        static Initializer initializer{ pargc, pargv };
        return initializer.initialization_status_;
    }

    Status Initializer::EnsureInitialized()
    {
        int argc = 0;
        char* argv_buf[] = { nullptr };
        char** argv = argv_buf;
        return EnsureInitialized(&argc, &argv);
    }

    Initializer::Initializer(int* pargc, char*** pargv)
        : initialization_status_{ Initialize(pargc, pargv) }
    {
    }

    bool InitLogSink(int* /*pargc*/, char*** /*pargv*/);

    Status Initializer::Initialize(int* pargc, char*** pargv)
    {
        try
        {
            Status status{};

            if (!pargc || !pargv) status = Status(LOTUS, StatusCode::INVALID_ARGUMENT);
            if (!status.IsOK()) return status;

            Lotus::InitLogSink(pargc, pargv);

            // LotusDeviceManager
            auto allocator_manager = AllocatorManager::Instance();
            if (!allocator_manager)
                return Status(LOTUS, StatusCode::FAIL, "Init allocator manager failed");

            status = allocator_manager->InitializeAllocators();
            if (status.IsOK())
                return status;

            return Status::OK();
        }
        catch (std::exception& ex)
        {
            return Status{ LOTUS, StatusCode::RUNTIME_EXCEPTION,
                std::string{ "Exception caught: " } +ex.what() };
        }
        catch (...)
        {
            return Status{ LOTUS, StatusCode::RUNTIME_EXCEPTION };
        }
    }
}