#include "core/common/status.h"

namespace Lotus {
namespace Common {
Status::Status(StatusCategory p_category, int p_code, const std::string& p_msg) {
  m_state.reset(new State());
  m_state->m_category = p_category;
  m_state->m_code = p_code;
  m_state->m_msg = p_msg;
}

Status::Status(StatusCategory p_category, int p_code)
    : Status(p_category, p_code, EmptyString()) {
}

bool Status::IsOK() const {
  return (m_state == NULL);
}

StatusCategory Status::Category() const {
  return IsOK() ? StatusCategory::NONE : m_state->m_category;
}

int Status::Code() const {
  return IsOK() ? static_cast<int>(StatusCode::OK) : m_state->m_code;
}

const std::string& Status::ErrorMessage() const {
  return IsOK() ? EmptyString() : m_state->m_msg;
}

std::string Status::ToString() const {
  if (m_state == nullptr) {
    return std::string("OK");
  }

  std::string result;

  if (StatusCategory::SYSTEM == m_state->m_category) {
    result += "SystemError";
    result += " : ";
    result += std::to_string(errno);
  } else if (StatusCategory::LOTUS == m_state->m_category) {
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
    result += m_state->m_msg;
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
