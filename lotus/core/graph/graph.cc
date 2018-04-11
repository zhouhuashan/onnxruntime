#include <fstream>
#include <iostream>
#include <numeric>
#include <stack>

#include "core/graph/graph.h"
#include "core/graph/op.h"
#include "core/graph/utils.h"

using namespace onnx::Utils;

namespace LotusIR {

#define NO_CHANGE_ON_SYNC_FLAG(...)              \
  do {                                           \
    bool sync_needed = graph_proto_sync_needed_; \
    { __VA_ARGS__; }                             \
    graph_proto_sync_needed_ = sync_needed;      \
  } while (0)

NodeArg::NodeArg(const std::string& name,
                 const TypeProto* p_node_arg_type) {
  node_arg_info_.set_name(name);
  // If the name is empty, it means the arg does not exist.
  exists_ = !(name.empty());
  if (nullptr != p_node_arg_type) {
    (*node_arg_info_.mutable_type()) = *p_node_arg_type;
    type_ = DataTypeUtils::ToType(node_arg_info_.type());
  } else {
    type_ = nullptr;
  }
}

const std::string& NodeArg::Name() const {
  return node_arg_info_.name();
}

const DataType NodeArg::Type() const {
  return type_;
}

const TensorShapeProto* NodeArg::Shape() const {
  if (!node_arg_info_.has_type()) {
    return nullptr;
  }

  auto typeCase = node_arg_info_.type().value_case();
  switch (typeCase) {
    case TypeProto::kTensorType: {
      if (node_arg_info_.type().tensor_type().has_shape()) {
        return &(node_arg_info_.type().tensor_type().shape());
      } else {
        return nullptr;
      }
    }
    case TypeProto::kSequenceType:
    case TypeProto::kMapType:
    case TypeProto::VALUE_NOT_SET:
    default:
      return nullptr;
  }
}

void NodeArg::SetShape(const TensorShapeProto& shape) {
  if (!node_arg_info_.has_type()) {
    return;
  }

  auto type_case = node_arg_info_.type().value_case();
  switch (type_case) {
    case TypeProto::kTensorType:
      *(node_arg_info_.mutable_type()->mutable_tensor_type()->mutable_shape()) = shape;
      break;
    case TypeProto::kSequenceType:
    case TypeProto::kMapType:
    case TypeProto::VALUE_NOT_SET:
    default:
      return;
  }
}

const NodeArgInfo& NodeArg::ToProto() const {
  return node_arg_info_;
}

void NodeArg::SetType(DataType p_type) {
  if (nullptr == p_type) {
    return;
  }

  type_ = p_type;
  *(node_arg_info_.mutable_type()) = DataTypeUtils::ToTypeProto(p_type);
}

void NodeArg::SetType(const TypeProto& type_proto) {
  type_ = DataTypeUtils::ToType(type_proto);
  *(node_arg_info_.mutable_type()) = type_proto;
}

bool NodeArg::Exists() const {
  return exists_;
}

Node::EdgeEnd::EdgeEnd(const Node* p_node, const NodeArg* p_node_arg)
    : node_(p_node), node_arg_(p_node_arg) {
}

const Node* Node::EdgeEnd::GetNode() const {
  return node_;
}

const NodeArg* Node::EdgeEnd::GetNodeArg() const {
  return node_arg_;
}

Node::NodeConstIterator::NodeConstIterator(
    std::set<const Node*>::const_iterator iter)
    : iter_(iter) {
}

bool Node::NodeConstIterator::operator==(
    const NodeConstIterator& other) const {
  return iter_ == other.iter_;
}

bool Node::NodeConstIterator::operator!=(
    const NodeConstIterator& other) const {
  return iter_ != other.iter_;
}

void Node::NodeConstIterator::operator++() {
  ++iter_;
}

const Node* Node::NodeConstIterator::operator*() {
  return *iter_;
}

Node::Node(const Node& other) {
  name_ = other.name_;
  op_type_ = other.op_type_;
  domain_ = other.domain_;
  input_defs_ = other.input_defs_;
  input_edges_ = other.input_edges_;
  output_edges_ = other.output_edges_;
  input_nodes_ = other.input_nodes_;
  control_inputs_ = other.control_inputs_;
  output_defs_ = other.output_defs_;
  output_nodes_ = other.output_nodes_;
  execution_provider_ = other.execution_provider_;
  attributes_ = other.attributes_;
}

NodeIndex Node::Index() const {
  return index_;
}

const std::string& Node::Name() const {
  return name_;
}

const std::string& Node::OpType() const {
  return op_type_;
}

const std::string& Node::Description() const {
  return description_;
}

const std::string& Node::Domain() const {
  return domain_;
}

const OpSchema* Node::Op() const {
  return op_;
}

const std::vector<NodeArg*>& Node::InputDefs() const {
  return input_defs_;
}

std::vector<NodeArg*>& Node::MutableInputDefs() {
  graph_->graph_resolve_needed_ = true;
  graph_->graph_proto_sync_needed_ = true;
  return input_defs_;
}

const std::vector<int>& Node::InputArgCount() const {
  return input_arg_count_;
}

std::vector<int>& Node::MutableInputArgCount() {
  graph_->graph_resolve_needed_ = true;
  graph_->graph_proto_sync_needed_ = true;
  return input_arg_count_;
}

Node::NodeConstIterator Node::InputNodesBegin() const {
  return NodeConstIterator(input_nodes_.begin());
}

Node::NodeConstIterator Node::InputNodesEnd() const {
  return NodeConstIterator(input_nodes_.end());
}

Node::NodeConstIterator Node::OutputNodesBegin() const {
  return NodeConstIterator(output_nodes_.begin());
}

Node::NodeConstIterator Node::OutputNodesEnd() const {
  return NodeConstIterator(output_nodes_.end());
}

const std::set<Node::EdgeEnd*>& Node::InputEdges() const {
  return input_edges_;
}

const std::set<Node::EdgeEnd*>& Node::OutputEdges() const {
  return output_edges_;
}

bool Node::InputEdgeSrcEnd(NodeArg* p_input_arg,
                           /*out*/ const EdgeEnd** pp_input_edge_src_end) const {
  if (nullptr == p_input_arg || nullptr == pp_input_edge_src_end) {
    return false;
  }

  for (const EdgeEnd* edge : input_edges_) {
    if (edge->GetNodeArg() == p_input_arg) {
      *pp_input_edge_src_end = edge;
      return true;
    }
  }

  return false;
}

const std::vector<NodeArg*>& Node::OutputDefs() const {
  return output_defs_;
}

std::vector<NodeArg*>& Node::MutableOutputDefs() {
  graph_->graph_resolve_needed_ = true;
  graph_->graph_proto_sync_needed_ = true;
  return output_defs_;
}

const std::string& Node::GetExecutionProvider() const {
  return execution_provider_;
}

void Node::SetExecutionProvider(const std::string& execution_provider) {
  execution_provider_ = execution_provider;
}

void Node::ToProto(NodeProto& proto) const {
  // Set name.
  proto.set_name(name_);
  // Set op type.
  proto.set_op_type(op_type_);
  // Set op domain;
  proto.set_domain(domain_);
  // Set doc string.
  proto.set_doc_string(description_);

  // Set attributes.
  proto.clear_attribute();
  for (auto attribute : attributes_) {
    auto attr = proto.add_attribute();
    *attr = attribute.second;
  }

  // Set inputs' definitions.
  proto.clear_input();
  for (auto& input_def : input_defs_) {
    auto input = proto.add_input();
    *input = input_def->Name();
  }

  // Set outputs' definitions.
  proto.clear_output();
  for (auto& output_def : output_defs_) {
    auto output = proto.add_output();
    *output = output_def->Name();
  }
}

void Node::Init(const NodeProto& node_proto,
                const ArgNameToTypeMap& name_to_type) {
  name_ = node_proto.name();
  op_type_ = node_proto.op_type();
  domain_ = node_proto.domain();

  for (int i = 0; i < node_proto.input().size(); ++i) {
    const TypeProto* type = nullptr;

    auto name_to_type_iter = name_to_type.find(node_proto.input(i));
    if (name_to_type.end() != name_to_type_iter) {
      // This node input arg type/shape does exist in graph proto.
      // Assign type/shape information to node input arg.
      type = &(name_to_type_iter->second);
    }

    NodeArg* node_arg = nullptr;
    auto node_arg_map = graph_->GetNodeArgMap();
    auto name_to_node_arg_iter = node_arg_map->find(node_proto.input(i));
    if (name_to_node_arg_iter == node_arg_map->end()) {
      node_arg = new NodeArg(node_proto.input(i), type);
      (*node_arg_map)[node_proto.input(i)] = node_arg;
    } else {
      node_arg = name_to_node_arg_iter->second;
    }

    input_defs_.push_back(node_arg);
  }

  // Set input arg count as 1:1 maping with input defs.
  // NOTE: it may be refined per operator definition.
  // There will be cases having arg count as, 1, 1, ..., 1, N.
  // It means that the last operator input is variadic.
  input_arg_count_.assign(input_defs_.size(), 1);

  for (int i = 0; i < node_proto.output().size(); ++i) {
    const TypeProto* type = nullptr;

    auto name_to_type_iter = name_to_type.find(node_proto.output(i));
    if (name_to_type.end() != name_to_type_iter) {
      // This output arg type/shape does exist in graph proto.
      // Assign type/shape information to node output arg.
      type = &(name_to_type_iter->second);
    }

    NodeArg* node_arg = nullptr;
    auto node_arg_map = graph_->GetNodeArgMap();
    auto name_to_node_arg_iter = node_arg_map->find(node_proto.output(i));
    if (name_to_node_arg_iter == node_arg_map->end()) {
      node_arg = new NodeArg(node_proto.output(i), type);
      (*node_arg_map)[node_proto.output(i)] = node_arg;
    } else {
      node_arg = name_to_node_arg_iter->second;
    }

    output_defs_.push_back(node_arg);
  }

  for (int i = 0; i < node_proto.attribute_size(); ++i) {
    auto& attr = node_proto.attribute(i);
    attributes_[attr.name()] = attr;
  }
}

void Node::Init(const std::string& name,
                const std::string& op_type,
                const std::string& description,
                const std::vector<NodeArg*>& input_args,
                const std::vector<NodeArg*>& output_args,
                const std::string& domain) {
  //Init(name, op_type, description, output_args, domain);
  name_ = name;
  op_type_ = op_type;
  description_ = description;
  output_defs_ = output_args;
  domain_ = domain;
  input_defs_ = input_args;

  // Set each arg count as 1 by default.
  // It could be adjusted when resolving the node with its operator
  // information.
  input_arg_count_.assign(input_defs_.size(), 1);

  auto node_arg_map = graph_->GetNodeArgMap();
  for (NodeArg* input_def : input_args) {
    auto name_to_node_arg_iter = node_arg_map->find(input_def->Name());
    if (name_to_node_arg_iter == node_arg_map->end()) {
      (*node_arg_map)[input_def->Name()] = input_def;
    } else {
      LOTUS_ENFORCE(name_to_node_arg_iter->second == input_def,
                    "Existing entry in NodeArg map for ", input_def->Name(), " != input definition.");
    }
  }

  for (NodeArg* output_def : output_args) {
    auto name_to_node_arg_iter = node_arg_map->find(output_def->Name());
    if (name_to_node_arg_iter == node_arg_map->end()) {
      (*node_arg_map)[output_def->Name()] = output_def;
    } else {
      LOTUS_ENFORCE(name_to_node_arg_iter->second == output_def,
                    "Existing entry in NodeArg map for ", output_def->Name(), " != input definition.");
    }
  }
}

/*void Node::Init(const std::string& name,
    const std::string& op_type,
    const std::string& description,
    const std::vector<NodeArg>& input_args,
    const std::vector<int>& p_inputArgCount,
    const std::vector<NodeArg>& output_args,
    const std::string& domain)
    {
    Init(name, op_type, description, output_args, domain);
    input_defs_ = input_args;
    input_arg_count_ = p_inputArgCount;
    }

    void Node::Init(const std::string& name,
    const std::string& op_type,
    const std::string& description,
    const std::vector<NodeArg>& output_args,
    const std::string& domain)
    {
    name_ = name;
    op_type_ = op_type;
    description_ = description;
    output_defs_ = output_args;
    domain_ = domain;
    }*/

void Node::AddAttribute(const std::string& attr_name, const AttributeProto& value) {
  graph_->graph_resolve_needed_ = true;
  graph_->graph_proto_sync_needed_ = true;
  attributes_[attr_name] = value;
}

#define ADD_BASIC_ATTR_IMPL(type, field)                                     \
  void Node::AddAttribute(const std::string& attr_name, const type& value) { \
    graph_->graph_resolve_needed_ = true;                                    \
    graph_->graph_proto_sync_needed_ = true;                                 \
    AttributeProto a;                                                        \
    a.set_name(attr_name);                                                   \
    a.set_##field(value);                                                    \
    attributes_[attr_name] = a;                                              \
  };

#define ADD_ATTR_IMPL(type, field)                                           \
  void Node::AddAttribute(const std::string& attr_name, const type& value) { \
    graph_->graph_resolve_needed_ = true;                                    \
    graph_->graph_proto_sync_needed_ = true;                                 \
    AttributeProto a;                                                        \
    a.set_name(attr_name);                                                   \
    *(a.mutable_##field()) = value;                                          \
    attributes_[attr_name] = a;                                              \
  };

#define ADD_LIST_ATTR_IMPL(type, field)                      \
  void Node::AddAttribute(const std::string& attr_name,      \
                          const std::vector<type>& values) { \
    graph_->graph_resolve_needed_ = true;                    \
    graph_->graph_proto_sync_needed_ = true;                 \
    AttributeProto a;                                        \
    a.set_name(attr_name);                                   \
    for (const auto& val : values) {                         \
      *(a.mutable_##field()->Add()) = val;                   \
    }                                                        \
    attributes_[attr_name] = a;                              \
  };

ADD_BASIC_ATTR_IMPL(float, f)
ADD_BASIC_ATTR_IMPL(int64_t, i)
ADD_BASIC_ATTR_IMPL(std::string, s)
ADD_ATTR_IMPL(TensorProto, t)
ADD_ATTR_IMPL(GraphProto, g)
ADD_LIST_ATTR_IMPL(float, floats)
ADD_LIST_ATTR_IMPL(int64_t, ints)
ADD_LIST_ATTR_IMPL(std::string, strings)
ADD_LIST_ATTR_IMPL(TensorProto, tensors)
ADD_LIST_ATTR_IMPL(GraphProto, graphs)

bool Node::ClearAttribute(const std::string& attr_name) {
  graph_->graph_resolve_needed_ = true;
  graph_->graph_proto_sync_needed_ = true;
  return attributes_.erase(attr_name) > 0;
}

const NodeAttributes& Node::GetAttributes() const {
  return attributes_;
}

Graph::Graph(const GraphProto& graph_proto,
             const std::unordered_map<std::string, int>& domain_to_version)
    : graph_proto_(graph_proto) {
  graph_proto_sync_needed_ = false;
  graph_resolve_needed_ = true;
  num_of_nodes_ = 0;

  domain_to_version_ = &domain_to_version;
  graph_type_ |= Type::Main;

  // Copy initial tensors to a map.
  for (auto tensor : graph_proto.initializer()) {
    name_to_initial_tensor_[tensor.name()] = tensor;
  }

  // Collect all node arg name, type, shape information in the graph.
  // type/shape information will be assigned to each node arg when going
  // thru all nodes later.
  ArgNameToTypeMap name_to_type_map;
  for (auto& graph_input : graph_proto_.input()) {
    if (graph_input.has_name() && graph_input.has_type()) {
      name_to_type_map[graph_input.name()] = graph_input.type();
    }
  }
  for (auto& graph_output : graph_proto_.output()) {
    if (graph_output.has_name() && graph_output.has_type()) {
      name_to_type_map[graph_output.name()] = graph_output.type();
    }
  }
  for (auto& node_arg : graph_proto_.value_info()) {
    if (node_arg.has_name() && node_arg.has_type()) {
      name_to_type_map[node_arg.name()] = node_arg.type();
    }
  }

  // Add nodes.
  AddSourceSinkNodes();
  for (auto node_proto : graph_proto.node()) {
    AddNode(node_proto, name_to_type_map);
  }
}

Graph::Graph(const std::string& name,
             const std::unordered_map<std::string, int>& domain_to_version) {
  graph_proto_sync_needed_ = false;
  graph_resolve_needed_ = true;
  num_of_nodes_ = 0;

  domain_to_version_ = &domain_to_version;
  graph_proto_.set_name(name);
  graph_type_ |= Type::Main;

  AddSourceSinkNodes();
}

Status Graph::VerifyNoDuplicateName(/*out*/ std::unordered_map<std::string, Node*>& output_args,
                                    /*out*/ std::unordered_map<std::string, NodeIndex>& node_name_to_index) {
  output_args.clear();
  node_name_to_index.clear();

  for (auto& node : Nodes()) {
    // Verify node name should be unique.
    auto& node_name = node.Name();

    if (!node_name.empty() && node_name_to_index.end() != node_name_to_index.find(node_name)) {
      // The node has name and its name was used by another node.
      Status status(LOTUS, FAIL,
                    "Error: two nodes with same node name (" + node_name + ").");
      return status;
    }

    node_name_to_index[node_name] = node.Index();

    // Verify node outputs' name should be unique.
    for (auto& output_def : node.OutputDefs()) {
      std::string output_arg_name = output_def->Name();
      if (output_args.end() != output_args.find(output_arg_name)) {
        // Two outputs with same name.
        Status status(LOTUS, FAIL,
                      "Error: two output args with same name (" + output_arg_name + ").");
        return status;
      }
      output_args.insert({output_arg_name, &node});
    }
  }
  return Status::OK();
}

Status Graph::BuildConnections(const std::unordered_map<std::string, Node*>& output_args,
                               const std::unordered_map<std::string, NodeIndex>& node_name_to_index) {
  std::unordered_set<Node*> inner_nodes;
  for (auto& node : Nodes()) {
    if (IsSourceNode(node.Index()) || IsSinkNode(node.Index())) {
      continue;
    }

    for (auto& control_input : node.control_inputs_) {
      auto name_to_index_iter = node_name_to_index.find(control_input);
      if (node_name_to_index.end() == name_to_index_iter) {
        Status status(LOTUS, FAIL,
                      "The control input (" + control_input + ") of Node (" +
                          node.Name() + ") does not exist in the graph.");
        return status;
      }

      NodeIndex src_node_index = name_to_index_iter->second;
      NodeIndex dst_node_index = node.Index();
      nodes_[src_node_index]->output_nodes_.insert(nodes_[dst_node_index].get());
      nodes_[dst_node_index]->input_nodes_.insert(nodes_[src_node_index].get());
    }

    auto& input_args = node.InputDefs();
    if (input_args.size() > 0) {
      // This node needs inputs.

      for (auto& input_arg : input_args) {
        if (!input_arg->Exists()) {
          // This input could be optional and it does not exist in this case.
          continue;
        }

        auto output_arg_iter = output_args.find(input_arg->Name());
        if (output_args.end() == output_arg_iter) {
          // No such output_arg matching this input_arg.
          // This input arg should be fed when running evaluation.

          // Add a control edge between <souce> node and this node.
          AddControlEdge(source_node_index_, node.Index());
          continue;
        }

        // Setup input/output relationship between <*node_iter>
        // and <output_arg_iter>.
        Node* output_node = output_arg_iter->second;

        node.input_nodes_.insert(output_node);
        Node::EdgeEnd* in_edge = new Node::EdgeEnd(output_node, input_arg);
        node.input_edges_.insert(in_edge);

        output_node->output_nodes_.insert(&node);
        Node::EdgeEnd* out_edge = new Node::EdgeEnd(&node, input_arg);
        output_node->output_edges_.insert(out_edge);

        inner_nodes.insert(output_node);
      }
    } else {
      if (node.OutputDefs().size() <= 0) {
        // This is a useless node.
        // It has no input/output.
        RemoveNode(node.Index());
      }

      // This is a starting node.
      // Add a control edge between <souce> node and this node.
      AddControlEdge(source_node_index_, node.Index());
    }
  }

  for (auto& node : Nodes()) {
    if (IsSourceNode(node.Index()) || IsSinkNode(node.Index())) {
      continue;
    }

    if (inner_nodes.size() <= 0 || inner_nodes.end() == inner_nodes.find(&node)) {
      // This is an ending node.
      // Add a control edge from this node to sink node.
      AddControlEdge(node.Index(), sink_node_index_);
    }
  }

  return Status::OK();
}

void Graph::ReverseDFSFrom(const std::vector<NodeIndex>& from,
                           const std::function<void(Node*)>& enter,
                           const std::function<void(Node*)>& leave,
                           const std::function<bool(const Node*, const Node*)>& comp) {
  std::vector<Node*> node_vec;
  for (auto i : from) {
    node_vec.push_back(GetNode(i));
  }

  ReverseDFSFrom(node_vec, enter, leave, comp);
}

void Graph::ReverseDFSFrom(const std::vector<Node*>& from,
                           const std::function<void(Node*)>& enter,
                           const std::function<void(Node*)>& leave,
                           const std::function<bool(const Node*, const Node*)>& comp) {
  using WorkEntry = std::pair<Node*, bool>;  // bool represents leave or not
  std::vector<WorkEntry> stack(from.size());
  for (size_t i = 0; i < from.size(); i++) {
    stack[i] = WorkEntry(from[i], false);
  }

  std::vector<bool> visited(nodes_.size(), false);
  while (!stack.empty()) {
    WorkEntry e = stack.back();
    stack.pop_back();
    Node* n = e.first;
    if (e.second) {
      // leave node
      leave(n);
      continue;
    }

    if (visited[n->Index()]) continue;
    visited[n->Index()] = true;

    if (enter) enter(n);

    if (leave) stack.push_back(WorkEntry(n, true));

    if (comp) {
      std::vector<Node*> sorted_nodes;
      for (auto iter = n->InputNodesBegin(); iter != n->InputNodesEnd(); ++iter) {
        sorted_nodes.push_back(const_cast<Node*>(*iter));
      }
      std::sort(sorted_nodes.begin(), sorted_nodes.end(), comp);
      for (auto in : sorted_nodes) {
        NodeIndex idx = in->Index();
        if (!visited[idx]) {
          stack.push_back(WorkEntry(in, false));
        }
      }
    } else {
      for (auto iter = n->InputNodesBegin(); iter != n->InputNodesEnd(); ++iter) {
        NodeIndex idx = (*iter)->Index();
        if (!visited[idx]) {
          stack.push_back(WorkEntry(GetNode(idx), false));
        }
      }
    }
  }
}

Status Graph::CheckIsAcyclic(std::vector<NodeIndex>& nodes_in_topological_order) {
  nodes_in_topological_order.clear();
  // nodes that have been processed and added to nodes_in_topological_order.
  std::unordered_set<NodeIndex> visited_nodes;
  std::unordered_set<NodeIndex> ancestor_nodes;
  // tracks nodes whose child nodes have been processed.
  std::unordered_set<NodeIndex> children_visited_nodes;
  std::stack<NodeIndex> stack;
  stack.push(sink_node_index_);

  while (!stack.empty()) {
    NodeIndex current = stack.top();
    stack.pop();

    if (visited_nodes.end() != visited_nodes.find(current)) {
      // The node has been visited before
      continue;
    }

    if (children_visited_nodes.end() != children_visited_nodes.find(current)) {
      // children are done so we mark this one complete.
      visited_nodes.insert(current);
      nodes_in_topological_order.push_back(current);
      ancestor_nodes.erase(current);
      continue;
    }

    if (nodes_[current]->InputNodesBegin() == nodes_[current]->InputNodesEnd()) {
      // no children
      children_visited_nodes.insert(current);
      visited_nodes.insert(current);
      nodes_in_topological_order.push_back(current);
      ancestor_nodes.erase(current);
      continue;
    }

    stack.push(current);

    // mark as children done. by the time the node is popped off the stack again,
    // its children will have been processed
    children_visited_nodes.insert(current);

    ancestor_nodes.insert(current);

    // check children
    for (auto iter = nodes_[current]->InputNodesBegin(); iter != nodes_[current]->InputNodesEnd(); ++iter) {
      NodeIndex idx = (*iter)->Index();
      if (ancestor_nodes.end() != ancestor_nodes.find(idx)) {
        Status status(LOTUS, FAIL, "Error: the graph is not acyclic.");
        return status;
      }

      // avoid re-processing nodes
      if (children_visited_nodes.end() == children_visited_nodes.find(idx)) {
        stack.push(idx);
      }
    }
  }

  if (this->NumberOfNodes() == nodes_in_topological_order.size()) {
    return Status::OK();
  } else {
    return Status(LOTUS, FAIL, "Error: the graph is not acyclic.");
  }
}

Status Graph::InferAndVerifyTypeMatch(Node* p_node,
                                      const OpSchema* p_op,
                                      const std::unordered_map<std::string, Node*>& output_args) {
  auto& nodeName = p_node->Name();

  // <k> index used to navigate node->InputDefs().
  int k = 0;
  std::unordered_map<std::string, DataType> type_parameter_to_type_map;

  for (size_t i = 0; i < p_node->InputArgCount().size(); ++i) {
    // Number of inputs matching to the i-th argument.
    int arg_count = p_node->InputArgCount()[i];
    // The i-th argument definition.
    auto op_formal_parameter = p_op->inputs()[i];

    // Infer and verify all <arguCount> inputs (k-th input)
    // matching operator definition (i-th argument).
    for (int j = 0; j < arg_count; ++j, ++k) {
      auto& input_def = p_node->MutableInputDefs()[k];

      // For each input arg.
      auto output_args_iter = output_args.find(input_def->Name());
      if (output_args.end() == output_args_iter) {
        // This input arg should either be fed by callers,
        // or be in initializers.
        // If it's fed by callers, it's needed to have type
        // information defined well.
        auto initial_tensor_iter = name_to_initial_tensor_.find(input_def->Name());
        if (name_to_initial_tensor_.end() != initial_tensor_iter) {
          // This input is fed with default value by initializer.
          // Infer its type from initializer tensor.
          TypeProto initial_tensor_type;
          initial_tensor_type.mutable_tensor_type()->set_elem_type(
              initial_tensor_iter->second.data_type());
          input_def->SetType(DataTypeUtils::ToType(initial_tensor_type));

          // Set shape accordingly.
          TensorShapeProto shape;
          for (auto dim : initial_tensor_iter->second.dims()) {
            shape.add_dim()->set_dim_value(dim);
          }
          input_def->SetShape(shape);
        } else if (!input_def->node_arg_info_.has_type()) {
          // This input is fed by callers and its type has to be specified.

          Status status(LOTUS, FAIL,
                        "Node (" + nodeName + ") input arg (" +
                            input_def->Name() + ") does not have type information.");
          return status;
        }
      } else {
        // The type of this input should have been set by
        // its parent who ouputs the NodeArg
        if (input_def->Type() == nullptr) {
          Status status(LOTUS, FAIL,
                        "Node (" + nodeName + ") input arg (" +
                            input_def->Name() + ") does not have type information set by parent node.");
          return status;
        }
      }

      // Verify the input arg type complying with operator
      // definition.

      auto iter = op_formal_parameter.GetTypes().find(input_def->Type());
      if (op_formal_parameter.GetTypes().end() == iter) {
        Status status(LOTUS, FAIL,
                      "Node (" + nodeName + ") input arg (" +
                          input_def->Name() + ") type does not match operator (" + p_op->Name() + ") definition.");
        return status;
      }

      auto param_to_type_iter = type_parameter_to_type_map.find(op_formal_parameter.GetTypeStr());
      if (type_parameter_to_type_map.end() == param_to_type_iter) {
        type_parameter_to_type_map[op_formal_parameter.GetTypeStr()] = input_def->Type();

      } else if (param_to_type_iter->second != input_def->Type() && arg_count == 1) {
        // This is the case.
        // An operator's inputs' type is "T", and T"s allowed value set is "float, int32".
        // However, one input is specified as "float", and another one is specified as "int".
        // NOTE: for variadic arguments (arg_count > 1), this verification rule is not applicable.
        // Different types are allowed for variadic arguments although there's only one type "T"
        // specified in op definition.
        Status status(LOTUS, FAIL,
                      "Node (" + nodeName + ") has different input types (" +
                          *(param_to_type_iter->second) + "," + *(input_def->Type()) +
                          ") matching to same type string (" + op_formal_parameter.GetTypeStr() + ").");
        return status;
      }
    }
  }

  // Infer and verify node output arg type information.
  int i = 0;
  for (auto& output_def : p_node->MutableOutputDefs()) {
    // For each output arg.

    auto op_formal_parameter = p_op->outputs()[i++];

    // Infer output arg type per input arg type if they share
    // the same type string. For example, type string is "T"
    // for both input arg and output arg.
    auto input_types_iter = type_parameter_to_type_map.find(op_formal_parameter.GetTypeStr());
    if (type_parameter_to_type_map.end() != input_types_iter) {
      output_def->SetType(input_types_iter->second);
      continue;
    }

    if (type_parameter_to_type_map.empty()) {
      // There's no input arg.
      // The output should be read from an attribute named kConstantValue.

      auto node_attributes_iter = p_node->GetAttributes().find(kConstantValue);
      if (p_node->GetAttributes().end() == node_attributes_iter) {
        Status status(LOTUS, FAIL,
                      "Node (" + nodeName + ") output arg value should be specified via node attribute '" +
                          kConstantValue + "'.");
        return status;
      }

      AttrType attr_type;
      RETURN_IF_ERROR(TypeUtils::GetType(node_attributes_iter->second, attr_type));

      if (AttrType::AttributeProto_AttributeType_TENSOR == attr_type) {
        auto& tensor = node_attributes_iter->second.t();
        TypeProto type_proto;
        type_proto.mutable_tensor_type()->set_elem_type(tensor.data_type());
        output_def->SetType(DataTypeUtils::ToType(type_proto));
      } else {
        Status status(LOTUS, FAIL,
                      "For attribute " + kConstantValue +
                          " , only Tensor type is allowed. The attribute type in this model is " +
                          LotusIR::kAttrTypeStrings[(int)attr_type] + ".");
        return status;
      }

      continue;
    }

    // For case that input arg and output arg have different types.
    if (output_def->node_arg_info_.has_type()) {
      // The output arg has already had type information.
      // Check whether it matches operator definition.
      auto iter = op_formal_parameter.GetTypes().find(output_def->Type());
      if (op_formal_parameter.GetTypes().end() == iter) {
        Status status(LOTUS, FAIL,
                      "Node (" + nodeName + ") output arg (" + output_def->Name() + ") type does not match operator (" + p_op->Name() + ") definition.");
        return status;
      }
      continue;
    }

    // Output arg has no type information.
    if (1 == op_formal_parameter.GetTypes().size()) {
      // Infer output arg type as the only one type defined
      // in operator definition.
      output_def->SetType(*(op_formal_parameter.GetTypes().begin()));
      continue;
    }

    // Output arg has no type information, and there're
    // multiple allowed types defined in operator definition.
    // Type inference fails in this case.
    Status status(LOTUS, FAIL,
                  "Node (" + nodeName + ") output arg (" + output_def->Name() + ") type inference failed");
    return status;
  }

  return Status::OK();
}

Status Graph::VerifyNodeAndOpMatch(
    const std::vector<NodeIndex>& nodes_in_topological_order,
    std::unordered_map<std::string, Node*>& output_args) {
  for (auto nodeIndex : nodes_in_topological_order) {
    if (IsSourceNode(nodeIndex) || IsSinkNode(nodeIndex)) {
      continue;
    }

    auto node = GetNode(nodeIndex);
    auto& node_name = node->Name();
    auto& op_type = node->OpType();
    auto& domain = node->Domain();
    auto version_iter = domain_to_version_->find(domain);
    if (domain_to_version_->end() == version_iter) {
      // The domain referred by this node does not exist either
      // in <OpSetIdProto> in the <ModelProto> loaded (in the case of model loaded from file) or
      // in global DomainToVersionRange map (in the case of model constructed from scratch).
      return Status(LOTUS, FAIL,
                    "The op domain (" + domain + ") used by node (" +
                        node_name + ") is not supported for this model.");
    }

    // Get op schema given op name, max inclusive version and domain.
    node->op_ = OpSchemaRegistry::Schema(op_type, version_iter->second, domain);
    if (nullptr == node->op_) {
      // A op_type refers to nothing.
      Status status(LOTUS, FAIL,
                    "Error: the operator or function (" + op_type + ") referred to by node (" +
                        node_name + ") does not exist.");
      return status;
    }

    auto op = node->Op();

    // The node refers to a primitive operator.
    // Infer and verify node input arg type information.
    auto total_arg_count = std::accumulate(node->InputArgCount().begin(),
                                           node->InputArgCount().end(), 0);

    if (total_arg_count != node->InputDefs().size()) {
      Status status(LOTUS, FAIL,
                    "The sum of input arg count is not equal to size of"
                    "input defs in node (" +
                        node_name + ").");
      return status;
    }

    // Verify size of node arg count is same as input number in
    // operator definition.
    if (op->inputs().size() != node->InputArgCount().size()) {
      // Adjust input arg count array with op definition
      // The adjustment will work as below,
      // In total, there're <total_arg_count> inputs, which
      // will be split as <1, 1, 1, 1, ... 1, x> or
      // <1, 1, 1, 1, ...1, 0, 0, ...0>. The final input
      // arg count array's element number will be the same
      // as op definition, and the sum of all elements will
      // be equal to <total_arg_count>.
      auto& input_arg_count = node->MutableInputArgCount();
      input_arg_count.clear();
      size_t m = 0;
      auto arg_count_left = total_arg_count;

      if (0 < op->inputs().size()) {
        for (; m < op->inputs().size() - 1; ++m) {
          if (arg_count_left > 0) {
            input_arg_count.push_back(1);
            arg_count_left--;
          } else {
            input_arg_count.push_back(0);
          }
        }
      }

      // Set the arg count for the last input formal parameter.
      // NOTE: in the case that there's no .input(...) defined
      // in op schema, all input args will be fed as one input
      // of the operator.
      input_arg_count.push_back(arg_count_left);
    }

    NO_CHANGE_ON_SYNC_FLAG(RETURN_IF_ERROR(InferAndVerifyTypeMatch(node, op, output_args)));

    // Attribute verification and fill node attribute with
    // default value defined in operator definition if needed.
    auto node_attributes = node->GetAttributes();
    for (auto attr_def : op->attributes()) {
      auto node_attr_iter = node_attributes.find(attr_def.first);
      if (node_attributes.end() == node_attr_iter) {
        if (!attr_def.second.required) {
          if (attr_def.second.default_value.has_name()) {
            // Set default value to the node attributes.
            node->AddAttribute(attr_def.first, attr_def.second.default_value);
          }
          // TODO: Handle optional attribute but no default value specified in op definition.
        } else {
          Status status(LOTUS, FAIL,
                        "Node (" + node_name + ") attribute (" + attr_def.first +
                            ") is required but not specified.");
          return status;
        }
      } else {
        // Verify node attribute type matching type of
        // attribute defined in operator definition.
        AttrType node_attr_type;
        RETURN_IF_ERROR(TypeUtils::GetType(node_attr_iter->second, node_attr_type));

        if (node_attr_type != attr_def.second.type) {
          Status status(LOTUS, FAIL,
                        "Node (" + node_name + ") attribute (" + node_attr_iter->first +
                            ") type does not match operator definition.");
          return status;
        }
      }
    }

    // Verify node with operator definition.
    NodeProto node_proto;
    node->ToProto(node_proto);
    try {
      node->Op()->Verify(node_proto);
    } catch (const std::exception& ex) {
      std::ostringstream errmsg;
      errmsg << "Exception throw while verifying node with schema with message: " << ex.what();
      return Status(LOTUS, FAIL, errmsg.str());
    }
  }

  return Status::OK();
}

Status Graph::Resolve() {
  if (!graph_resolve_needed_) {
    return Status::OK();
  }

  std::unordered_map<std::string, Node*> output_args;
  std::unordered_map<std::string, NodeIndex> node_name_to_index;
  RETURN_IF_ERROR(VerifyNoDuplicateName(output_args, node_name_to_index));
  RETURN_IF_ERROR(BuildConnections(output_args, node_name_to_index));
  RETURN_IF_ERROR(CheckIsAcyclic(nodes_in_topological_order_));
  RETURN_IF_ERROR(VerifyNodeAndOpMatch(nodes_in_topological_order_, output_args));
  RETURN_IF_ERROR(SetGraphInputsOutputs());

  graph_resolve_needed_ = false;
  return Status::OK();
}

Status GraphBase::GetNodesInTopologicalOrder(const std::vector<NodeIndex>** pp_nodes) {
  RETURN_IF_ERROR(Resolve());

  *pp_nodes = &nodes_in_topological_order_;
  return Status::OK();
}

void GraphBase::AddSourceSinkNodes() {
  std::vector<NodeArg*> empty_args;

  source_node_index_ = AddNode("_Graph_Source", kNoOp,
                               "Source node internally in a graph.", empty_args, empty_args)
                           ->Index();

  sink_node_index_ = AddNode("_Graph_Sink", kNoOp,
                             "Sink node internally in a graph.", empty_args, empty_args)
                         ->Index();

  AddControlEdge(source_node_index_, sink_node_index_);
}

const std::string& Graph::Name() const {
  return graph_proto_.name();
}

void Graph::SetName(const std::string& name) {
  graph_proto_.set_name(name);
}

const std::string& Graph::Description() const {
  return graph_proto_.doc_string();
}

void Graph::SetDescription(const std::string& description) {
  graph_proto_.set_doc_string(description);
}

std::unordered_map<std::string, NodeArg*>* Graph::GetNodeArgMap() {
  return &node_args_;
}

void Graph::AddInitializedTensor(const TensorProto& tensor) {
  name_to_initial_tensor_[tensor.name()] = tensor;
  graph_proto_sync_needed_ = true;
  graph_resolve_needed_ = true;
}

void Graph::RemoveInitializedTensor(const std::string& tensor_name) {
  name_to_initial_tensor_.erase(tensor_name);
  graph_proto_sync_needed_ = true;
  graph_resolve_needed_ = true;
}

bool Graph::GetInitializedTensor(const std::string& tensor_name, TensorProto& value) const {
  auto iter = name_to_initial_tensor_.find(tensor_name);
  if (name_to_initial_tensor_.end() == iter) {
    return false;
  }
  value = iter->second;
  return true;
}

void Graph::CleanAllInitializedTensors() {
  name_to_initial_tensor_.clear();
}

const InitializedTensorSet& Graph::GetAllInitializedTensors() const {
  return name_to_initial_tensor_;
}

const std::vector<const NodeArg*>& Graph::GetInputs() const {
  return graph_inputs_;
}

const std::vector<const NodeArg*>& Graph::GetOutputs() const {
  return graph_outputs_;
}

const std::vector<const NodeArg*>& Graph::GetValueInfo() const {
  return value_info_;
}

Node* GraphBase::GetNode(NodeIndex node_index) {
  if (MaxNodeIndex() <= node_index) {
    return nullptr;
  }

  return nodes_[node_index].get();
}

const Node* GraphBase::GetNode(NodeIndex node_index) const {
  if (MaxNodeIndex() <= node_index) {
    return nullptr;
  }

  return nodes_[node_index].get();
}

NodeIndex GraphBase::MaxNodeIndex() const {
  return nodes_.size();
}

int GraphBase::NumberOfNodes() const {
  return num_of_nodes_;
}

Node* GraphBase::AddNode(const NodeProto& node_proto,
                         const ArgNameToTypeMap& name_to_type_map) {
  auto node = AllocateNode();
  node->Init(node_proto, name_to_type_map);
  return node;
}

Node* GraphBase::AddNode(const std::string& name,
                         const std::string& op_type,
                         const std::string& description,
                         const std::vector<NodeArg*>& input_args,
                         const std::vector<NodeArg*>& output_args,
                         const std::string& domain) {
  auto node = AllocateNode();
  node->Init(name, op_type, description, input_args, output_args, domain);
  if (0 != op_type.compare(kNoOp)) {
    graph_proto_sync_needed_ = true;
  }
  return node;
}

/*Node* GraphBase::AddNode(const std::string& name,
    const std::string& op_type,
    const std::string& description,
    const std::vector<NodeArg>& input_args,
    const std::vector<int>& p_inputArgCount,
    const std::vector<NodeArg>& output_args,
    const std::string& domain)
    {
    auto node = AllocateNode();
    node->Init(name,
    op_type,
    description,
    input_args,
    p_inputArgCount,
    output_args,
    domain);
    graph_proto_sync_needed_  = true;
    return node;
    }*/

/*Node* GraphBase::AddNode(const std::string& name,
    const std::string& op_type,
    const std::string& description,
    const std::vector<NodeArg>& output_args,
    const std::string& domain)
    {
    auto node = AllocateNode();
    node->Init(name,
    op_type,
    description,
    output_args,
    domain);
    graph_proto_sync_needed_  = true;
    return node;
    }*/

Node* GraphBase::AddNode(const Node& other) {
  auto node = AllocateNode();
  *node = other;
  graph_proto_sync_needed_ = true;
  return node;
}

bool GraphBase::RemoveNode(NodeIndex p_index) {
  if (MaxNodeIndex() <= p_index || nullptr == nodes_[p_index]) {
    return false;
  }

  ReleaseNode(p_index);
  return true;
}

Node* GraphBase::AddConstantNode(const std::string& name,
                                 const std::string& description,
                                 const std::vector<NodeArg*>& output_args,
                                 const TensorProto& tensor) {
  Node* node = AddNode(name, kConstant, description, std::vector<NodeArg*>{}, output_args);
  node->AddAttribute(kConstantValue, tensor);
  return node;
}

bool GraphBase::AddControlEdge(NodeIndex src_node_index, NodeIndex dst_node_index) {
  if (MaxNodeIndex() <= src_node_index || MaxNodeIndex() <= dst_node_index ||
      nullptr == nodes_[src_node_index] || nullptr == nodes_[dst_node_index]) {
    // Invalid node indexes specified.
    return false;
  }
  nodes_[src_node_index]->output_nodes_.insert(nodes_[dst_node_index].get());
  nodes_[dst_node_index]->input_nodes_.insert(nodes_[src_node_index].get());
  nodes_[dst_node_index]->control_inputs_.insert(nodes_[src_node_index]->Name());

  if (!IsSourceNode(src_node_index) && !IsSinkNode(dst_node_index)) {
    graph_proto_sync_needed_ = true;
    graph_resolve_needed_ = true;
  }

  return true;
}

const GraphProto& Graph::ToGraphProto() {
  if (!graph_proto_sync_needed_) {
    return graph_proto_;
  }

  // Nodes.
  graph_proto_.clear_node();

  // Nodes must be sorted in Topological Order in the GraphProto per ONNX spec.
  for (auto& node_idx : nodes_in_topological_order_) {
    if (IsSourceNode(node_idx) || IsSinkNode(node_idx)) {
      continue;
    }
    auto node_proto = graph_proto_.add_node();
    nodes_[node_idx]->ToProto(*node_proto);
  }

  // Initial tensors;
  graph_proto_.clear_initializer();
  for (auto item : name_to_initial_tensor_) {
    auto tensor = graph_proto_.add_initializer();
    *tensor = item.second;
  }

  // Sync graph inputs/outputs/valueInfo.
  SyncGraphInputsOutputs();

  graph_proto_sync_needed_ = false;

  return graph_proto_;
}

void Graph::SyncGraphInputsOutputs() {
  graph_proto_.clear_input();
  graph_proto_.clear_output();
  graph_proto_.clear_value_info();

  for (auto inputArg : graph_inputs_) {
    *(graph_proto_.mutable_input()->Add()) = inputArg->ToProto();
  }

  for (auto outputArg : graph_outputs_) {
    *(graph_proto_.mutable_output()->Add()) = outputArg->ToProto();
  }

  for (auto valueInfo : value_info_) {
    *(graph_proto_.mutable_value_info()->Add()) = valueInfo->ToProto();
  }
}

Status Graph::SetGraphInputsOutputs() {
  // Reset graphInputs/graphOutputs/valueInfo state.
  graph_inputs_.clear();
  graph_outputs_.clear();
  value_info_.clear();

  // Flag indicates that this graph is loaded from model file.
  // If it's true, then graph inputs and outputs will keep the same
  // as what are specified in the model, otherwise, graph inputs
  // and outputs will be inferred.
  bool loaded_from_model_file = graph_proto_.input_size() != 0 ||
                                graph_proto_.output_size() != 0 ||
                                graph_proto_.value_info_size() != 0;

  std::unordered_set<std::string> added_input_names{};

  if (loaded_from_model_file) {
    // Collect all graph inputs/outputs specified in original graph proto
    std::unordered_set<std::string> specified_graph_inputs;
    std::unordered_set<std::string> specified_graph_outputs;
    std::unordered_set<std::string> specified_graph_value_info;
    std::unordered_set<std::string> specified_initializers;

    for (auto& graph_input : graph_proto_.input()) {
      specified_graph_inputs.insert(graph_input.name());
    }

    for (auto& graph_output : graph_proto_.output()) {
      specified_graph_outputs.insert(graph_output.name());
    }

    for (auto& graph_value_info : graph_proto_.value_info()) {
      specified_graph_value_info.insert(graph_value_info.name());
    }

    for (auto& initializer : graph_proto_.initializer()) {
      specified_initializers.insert(initializer.name());
    }

    std::unordered_map<std::string, const NodeArg*> output_name_to_node_arg;
    for (auto& node : Nodes()) {
      for (auto& output_def : node.OutputDefs()) {
        if (specified_graph_outputs.erase(output_def->Name()) >= 1) {
          graph_outputs_.push_back(output_def);
        }
        output_name_to_node_arg.insert({output_def->Name(), output_def});
      }
    }

    if (specified_graph_outputs.size() != 0) {
      return Status(LOTUS, FAIL, "Some graph outputs which don't exist in the graph.");
    }

    for (auto& node : Nodes()) {
      // Go thru all node's inputs.
      for (auto& input_arg : node.InputDefs()) {
        if (!input_arg->Exists()) {
          // It's an optional input and does not exist in this case.
          continue;
        }

        if (specified_graph_inputs.end() != specified_graph_inputs.find(input_arg->Name())) {
          if (added_input_names.end() == added_input_names.find(input_arg->Name())) {
            // The node input is specified as graph input.
            graph_inputs_.push_back(input_arg);
            added_input_names.insert(input_arg->Name());
          }
          continue;
        }

        auto output_arg_iter = output_name_to_node_arg.find(input_arg->Name());
        if (output_name_to_node_arg.end() == output_arg_iter && specified_initializers.end() == specified_initializers.find(input_arg->Name())) {
          // The node input is not specified as graph input,
          // and it's not fed by another node neither.
          return Status(LOTUS, FAIL, "Node input (" + input_arg->Name() + ") should be a graph input.");
        }

        if (specified_graph_value_info.erase(input_arg->Name()) >= 1) {
          value_info_.push_back(input_arg);
        }
      }
    }
  } else {
    std::unordered_map<std::string, const NodeArg*> output_name_to_node_arg;
    for (auto& node : Nodes()) {
      for (auto& output_def : node.OutputDefs()) {
        output_name_to_node_arg.insert({output_def->Name(), output_def});
      }
    }

    // Init graph output args with all node output args.
    auto graph_output_args = output_name_to_node_arg;

    std::unordered_set<Node*> inner_nodes;
    for (auto& node : Nodes()) {
      // Go thru all node's inputs.
      for (auto& input_arg : node.InputDefs()) {
        if (!input_arg->Exists()) {
          // It's an optional input and does not exist in this case.
          continue;
        }

        auto output_arg_iter = output_name_to_node_arg.find(input_arg->Name());
        if (output_name_to_node_arg.end() == output_arg_iter) {
          // This input arg should be fed when running evaluation.
          // it should be a graph input.
          if (added_input_names.end() == added_input_names.find(input_arg->Name())) {
            // This graph input has not been added into <graph_inputs_>.
            graph_inputs_.push_back(input_arg);
            added_input_names.insert(input_arg->Name());
          }
        } else if (graph_output_args.erase(output_arg_iter->first) >= 1) {
          // Remove the output arg name from graph outputs since it's
          // the input of another node, which we call it intermediate result
          // and store it in <m_valueinfo>.
          value_info_.push_back(input_arg);
        }
      }
    }

    // Set graph outputs.
    for (auto& output_arg : graph_output_args) {
      graph_outputs_.push_back(output_arg.second);
    }
  }

  return Status::OK();
}

bool GraphBase::IsSourceNode(NodeIndex index) const {
  return source_node_index_ == index;
}

bool GraphBase::IsSinkNode(NodeIndex index) const {
  return sink_node_index_ == index;
}

const Node* GraphBase::SourceNode() const {
  return nodes_[source_node_index_].get();
}

const Node* GraphBase::SinkNode() const {
  return nodes_[sink_node_index_].get();
}

Node* GraphBase::AllocateNode() {
  std::unique_ptr<Node> node(new Node(MaxNodeIndex(), this));
  nodes_.push_back(std::move(node));
  num_of_nodes_++;
  graph_resolve_needed_ = true;
  return nodes_.back().get();
}

void GraphBase::ReleaseNode(NodeIndex index) {
  nodes_[index] = nullptr;
  num_of_nodes_--;
  graph_proto_sync_needed_ = true;
  graph_resolve_needed_ = true;
}
}  // namespace LotusIR
