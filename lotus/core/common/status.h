#pragma once

#include <memory>
#include <string>

namespace Lotus {
namespace Common {

#define RETURN_IF_ERROR(expr)              \
  do {                                     \
    auto _status = (expr);                 \
    if ((!_status.IsOK())) return _status; \
  } while (0)

enum StatusCategory {
  NONE = 0,
  SYSTEM = 1,
  LOTUS = 2,
};

// Error code for lotus.
enum StatusCode {
  OK = 0,
  FAIL = 1,
  INVALID_ARGUMENT = 2,
  NO_SUCHFILE = 3,
  NO_MODEL = 4,
  ENGINE_ERROR = 5,
  RUNTIME_EXCEPTION = 6,
  INVALID_PROTOBUF = 7,
  MODEL_LOADED = 8,
  NOT_IMPLEMENTED = 9,
};

class Status {
 public:
  Status() {}

  Status(StatusCategory category, int code, const std::string& msg);

  Status(StatusCategory category, int code);

  inline Status(const Status& other)
      : state_((other.state_ == nullptr) ? nullptr : std::make_unique<State>(*other.state_)) {}

  bool IsOK() const;

  int Code() const;

  StatusCategory Category() const;

  const std::string& ErrorMessage() const;

  std::string ToString() const;

  inline void operator=(const Status& other) {
    if (&other != this) {
      if (nullptr == other.state_) {
        state_.reset();
      } else if (state_ != other.state_) {
        state_.reset(new State(*other.state_));
      }
    }
  }

  inline bool operator==(const Status& other) const {
    return (this->state_ == other.state_) || (ToString() == other.ToString());
  }

  inline bool operator!=(const Status& other) const {
    return !(*this == other);
  }

  static const Status& OK();

 private:
  static const std::string& EmptyString();

  struct State {
    StatusCategory category_;
    int code_;
    std::string msg_;
  };

  // As long as Code() is OK, state_ == nullptr.
  std::unique_ptr<State> state_;
};
}  // namespace Common
}  // namespace Lotus
