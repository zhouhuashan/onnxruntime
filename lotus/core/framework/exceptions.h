#ifndef CORE_FRAMEWORK_EXCEPTIONS_H
#define CORE_FRAMEWORK_EXCEPTIONS_H
#include <stdexcept>

namespace Lotus {
    class NotImplementedException : public std::logic_error
    {
    public:
        NotImplementedException() : std::logic_error("Function not yet implemented") { };

    };
}

#endif