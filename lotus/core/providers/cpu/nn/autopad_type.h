#ifndef CORE_PROVIDERS_CPU_NN_AUTOPAD_TYPE_H
#define CORE_PROVIDERS_CPU_NN_AUTOPAD_TYPE_H
#include "core/common/common.h"
#include "core/common/exceptions.h"

namespace Lotus {

enum class AutoPadType {
  NOTSET = 0,
  VALID = 1,
  SAME_UPPER = 2,
  SAME_LOWER = 3,
};

inline AutoPadType StringToAutoPadType(const string& str) {
  if (str.empty()) {
    return AutoPadType::NOTSET;
  } else if (str == "VALID") {
    return AutoPadType::VALID;
  } else if (str == "SAME_UPPER") {
    return AutoPadType::SAME_UPPER;
  } else if (str == "SAME_LOWER") {
    return AutoPadType::SAME_LOWER;
  } else {
    LOTUS_ENFORCE(false, "Unknown AutoPadType String");
  }
}
}  // namespace Lotus

#endif  // CORE_PROVIDERS_CPU_NN_AUTOPAD_TYPE_H
