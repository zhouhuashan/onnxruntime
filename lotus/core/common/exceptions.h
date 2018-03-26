#pragma once

#include <exception>
#include <stdexcept>
#include "core/common/common.h"

namespace Lotus {

class NotImplementedException : public std::logic_error {
 public:
  NotImplementedException() : std::logic_error("Function not yet implemented"){};
};

class TypeMismatchException : public std::logic_error {
 public:
  TypeMismatchException() : logic_error("Type mismatch"){};
};

class EnforceNotMet : public std::exception {
 public:
  EnforceNotMet(
      const char* file,
      const int line,
      const char* condition,
      const std::string& msg,
      const void* caller = nullptr);
  void AppendMessage(const std::string& msg);
  std::string Msg() const;
  inline const std::vector<std::string>& MsgStack() const {
    return msg_stack_;
  }

  const char* what() const noexcept override;

  const void* Caller() const noexcept;

 private:
  std::vector<std::string> msg_stack_;
  std::string full_msg_;
  std::string stack_trace_;
  const void* caller_;
};
}  // namespace Lotus
