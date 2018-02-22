#ifndef CORE_FRAMEWORK_EXCEPTIONS_H
#define CORE_FRAMEWORK_EXCEPTIONS_H
#include <stdexcept>
#include <exception>
#include "core/common/common.h"

namespace Lotus {

    class NotImplementedException : public std::logic_error
    {
    public:
        NotImplementedException() : std::logic_error("Function not yet implemented") { };

    };

    class TypeMismatchException : public std::logic_error
    {
    public:
        TypeMismatchException() : logic_error("Type mismatch") {};
    };

    class EnforceNotMet : public std::exception {
    public:
        EnforceNotMet(
            const char* file,
            const int line,
            const char* condition,
            const string& msg,
            const void* caller = nullptr);
        void AppendMessage(const string& msg);
        string msg() const;
        inline const vector<string>& msg_stack() const {
            return msg_stack_;
        }

        const char* what() const noexcept override;

        const void* caller() const noexcept;

    private:
        vector<string> msg_stack_;
        string full_msg_;
        string stack_trace_;
        const void* caller_;
    };
}

#endif