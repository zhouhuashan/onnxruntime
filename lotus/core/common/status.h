#pragma once

#include <memory>
#include <string>
#include "core/inc/ml_status.h"

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
  OK = MLStatus::OK,
  FAIL = MLStatus::FAIL,
  INVALID_ARGUMENT = MLStatus::INVALID_ARGUMENT,
  NO_SUCHFILE = MLStatus::NO_SUCHFILE,
  NO_MODEL = MLStatus::NO_MODEL,
  ENGINE_ERROR = MLStatus::ENGINE_ERROR,
  RUNTIME_EXCEPTION = MLStatus::RUNTIME_EXCEPTION,
  INVALID_PROTOBUF = MLStatus::INVALID_PROTOBUF,
  MODEL_LOADED = MLStatus::MODEL_LOADED,
  NOT_IMPLEMENTED = MLStatus::NOT_IMPLEMENTED,
};

class Status {
 public:
  Status() noexcept {}

  Status(StatusCategory category, int code, const std::string& msg);

  Status(StatusCategory category, int code);

  Status(const Status& other)
      : state_((other.state_ == nullptr) ? nullptr : std::make_unique<State>(*other.state_)) {}

  void operator=(const Status& other) {
    if (&other != this) {
      if (nullptr == other.state_) {
        state_.reset();
      } else if (state_ != other.state_) {
        state_ = std::make_unique<State>(*other.state_);
      }
    }
  }

  Status(Status&& other) = default;
  Status& operator=(Status&& other) = default;
  ~Status() = default;

  bool IsOK() const noexcept;

  int Code() const noexcept;

  StatusCategory Category() const noexcept;

  const std::string& ErrorMessage() const;

  std::string ToString() const;

  bool operator==(const Status& other) const {
    return (this->state_ == other.state_) || (ToString() == other.ToString());
  }

  bool operator!=(const Status& other) const {
    return !(*this == other);
  }

  static const Status& OK() noexcept;

 private:
  static const std::string& EmptyString();

  struct State {
    State(StatusCategory cat0, int code0, const std::string& msg0) : category(cat0), code(code0), msg(msg0) {}

    StatusCategory category = StatusCategory::NONE;
    int code = 0;
    std::string msg;
  };

  // As long as Code() is OK, state_ == nullptr.
  std::unique_ptr<State> state_;
};

inline std::ostream& operator<<(std::ostream& out, const Status& status) {
  return out << status.ToString();
}

}  // namespace Common
}  // namespace Lotus
