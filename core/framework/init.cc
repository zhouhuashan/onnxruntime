#include "core/framework/init.h"
#include "core/framework/allocatormgr.h"

namespace Lotus {

    Common::Status Initializer::EnsureInitialized(int* pargc, char*** pargv)
    {
        static Initializer initializer{ pargc, pargv };
        return initializer.initialization_status_;
    }

    Common::Status Initializer::EnsureInitialized()
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

    Common::Status Initializer::Initialize(int* pargc, char*** pargv)
    {
        try
        {
            Common::Status status{};

            if (!pargc || !pargv) status = Common::Status(Common::LOTUS, Common::StatusCode::INVALID_ARGUMENT);
            if (!status.Ok()) return status;

            // LotusDeviceManager
            auto allocator_manager = AllocatorManager::Instance();
            if (!allocator_manager)
                return Common::Status(Common::LOTUS, Common::StatusCode::FAIL, "Init allocator manager failed");

            status = allocator_manager->InitializeAllocators();
            if (status.Ok())
                return status;

            return Common::Status::OK();
        }
        catch (std::exception& ex)
        {
            return Status{ Common::LOTUS, Common::StatusCode::RUNTIME_EXCEPTION,
                std::string{ "Exception caught: " } +ex.what() };
        }
        catch (...)
        {
            return Status{ Common::LOTUS, Common::StatusCode::RUNTIME_EXCEPTION };
        }
    }
}