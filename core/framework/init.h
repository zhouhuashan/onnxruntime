#ifndef CORE_FRAMEWORK_INIT_H
#define CORE_FRAMEWORK_INIT_H
#include "core\graph\status.h"

namespace Lotus {
    /**
    * Performs Lotus initialization logic at most once.
    */
    class Initializer
    {
    public:
        Initializer(const Initializer&) = delete;
        Initializer& operator=(const Initializer&) = delete;

        /**
        * Runs the initialization logic if it hasn't been run yet.
        */
        static Common::Status EnsureInitialized(int* pargc, char*** pargv);
        static Common::Status EnsureInitialized();

    private:
        Initializer(int* pargc, char*** pargv);
        Common::Status Initialize(int* pargc, char*** pargv);

        Common::Status initialization_status_;
    };
}

#endif