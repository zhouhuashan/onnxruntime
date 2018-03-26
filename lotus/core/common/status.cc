#include "core/common/status.h"

namespace Lotus {
namespace Common {
Status::Status(StatusCategory category, int code, const std::string& msg) {
  state_.reset(new State());
  state_->category_ = category;
  state_->code_ = code;
  state_->msg_ = msg;
}

Status::Status(StatusCategory category, int code)
    : Status(category, code, EmptyString()) {
}

bool Status::IsOK() const {
  return (state_ == NULL);
}

StatusCategory Status::Category() const {
  return IsOK() ? StatusCategory::NONE : state_->category_;
}

int Status::Code() const {
  return IsOK() ? static_cast<int>(StatusCode::OK) : state_->code_;
}

const std::string& Status::ErrorMessage() const {
  return IsOK() ? EmptyString() : state_->msg_;
}

std::string Status::ToString() const {
  if (state_ == nullptr) {
    return std::string("OK");
  }

  std::string result;

  if (StatusCategory::SYSTEM == state_->category_) {
    result += "SystemError";
    result += " : ";
    result += std::to_string(errno);
  } else if (StatusCategory::LOTUS == state_->category_) {
    result += "[LotusError]";
    result += " : ";
    result += std::to_string(static_cast<int>(Code()));
    std::string msg;
    switch (static_cast<StatusCode>(Code())) {
      case INVALID_ARGUMENT:
        msg = "INVALID_ARGUMENT";
        break;
      case NO_SUCHFILE:
        msg = "NO_SUCHFILE";
        break;
      case NO_MODEL:
        msg = "NO_MODEL";
        break;
      case ENGINE_ERROR:
        msg = "ENGINE_ERROR";
        break;
      case RUNTIME_EXCEPTION:
        msg = "RUNTIME_EXCEPTION";
        break;
      case INVALID_PROTOBUF:
        msg = "INVALID_PROTOBUF";
        break;
      case MODEL_LOADED:
        msg = "MODEL_LOADED";
        break;
      case NOT_IMPLEMENTED:
        msg = "NOT_IMPLEMENTED";
        break;
      default:
        msg = "GENERAL ERROR";
        break;
    }
    result += " : ";
    result += msg;
    result += " : ";
    result += state_->msg_;
  }

  return result;
}

const Status& Status::OK() {
  static Status s_ok;
  return s_ok;
}

const std::string& Status::EmptyString() {
  static std::string s_emptyStr = "";
  return s_emptyStr;
}
}  // namespace Common
}  // namespace Lotus
