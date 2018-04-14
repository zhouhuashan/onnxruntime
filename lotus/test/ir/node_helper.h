#pragma once

#include "core/graph/graph.h"

namespace LotusIR {
namespace Test {

struct NodeTestHelper {
  // helper to provide mutable access to the Node internal definitions
  static Node::Definitions& MutableDefinitions(Node& node) {
    return node.MutableDefinitions();
  }

  // helper to provide mutable access to the Node internal relationships
  static Node::Relationships& MutableRelationships(Node& node) {
    return node.MutableRelationships();
  }
};
}  // namespace Test
}  // namespace LotusIR
