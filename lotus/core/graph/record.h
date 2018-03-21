#ifndef LOTUSIR_CORE_GRAPH_RECORD_H
#define LOTUSIR_CORE_GRAPH_RECORD_H

#include <assert.h>
#include <string>
#include <tuple>
#include <vector>

#include "core/common/status.h"

namespace Lotus {
namespace Common {
template <class... Types>
class Record {
 public:
  typedef std::tuple<Types...> VALUES;

  Record() = default;

  Record(const std::vector<std::string>& p_names,
         const VALUES& p_values) {
    assert(std::tuple_size<VALUES>::value == p_names.size());
    m_names = p_names;
    m_values = p_values;
  }

  Record(const Record<Types...>& p_other) {
    m_names = p_other.m_names;
    m_values = p_other.m_values;
  }

  Status GetName(int p_index, const std::string** p_name) const {
    if (nullptr == p_name || p_index >= m_names.size()) {
      return Status(LOTUS, StatusCode::INVALID_ARGUMENT);
    }

    *p_name = &(m_names[p_index]);
    return Status::OK();
  }

  const VALUES& GetValues() const {
    return m_values;
  }

 private:
  std::vector<std::string> m_names;

  VALUES m_values;
};
}  // namespace Common
}  // namespace Lotus

#endif