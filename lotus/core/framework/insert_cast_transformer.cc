#include "core/framework/insert_cast_transformer.h"
#include "core/framework/data_types.h"
#include "core/inc/op_kernel_author.h"
#include "core/framework/kernel_registry.h"

using namespace onnx;
using namespace ::Lotus::Common;
namespace Lotus {
class IdGenerator {
 public:
  int Next() {
    return id++;
  }

 private:
  int id = 0;
};

bool InsertCastTransformer::NeedInsertCast(const LotusIR::Node* node, const LotusIR::NodeArg* input) const {
  //If the node's input is float16 and currently the node is not assigned to any XP.
  //we need insert a cast to float, and put the node on CPU for default behavior.
  //TODO: a better check is to check does the CPU kernel with float exist or not.
  if (input->Type() != nullptr &&
      DataTypeImpl::TypeFromProto(*input->TypeAsProto()) == DataTypeImpl::GetTensorType<MLFloat16>() &&
      node->GetExecutionProviderType().empty()) {
    return true;
  }
  return false;
}

LotusIR::NodeArg* AddCastNode(LotusIR::Graph& graph,
                              IdGenerator& id_generator,
                              LotusIR::NodeArg* old_arg,
                              TypeProto* new_type,
                              bool new_on_input,
                              int64_t to_type,
                              LotusIR::ProviderType providerType) {
  //insert cast op to cast input
  int id = id_generator.Next();

  char str[32];
  snprintf(str, 32, "CastDef_%d", id);

  auto* new_arg = &graph.GetOrCreateNodeArg(str, new_type);

  std::vector<LotusIR::NodeArg*> input_defs = {new_on_input ? new_arg : old_arg};
  std::vector<LotusIR::NodeArg*> output_defs = {new_on_input ? old_arg : new_arg};

  auto cast_node = graph.AddNode(str, "Cast", "cast node to cast from float16 to float32 on cpu", input_defs, output_defs);
  cast_node->AddAttribute("to", to_type);
  cast_node->SetExecutionProviderType(providerType);
  return new_arg;
}

Status InsertCastTransformer::Apply(LotusIR::Graph& graph, bool& modified) const {
  LOTUS_RETURN_IF_ERROR(graph.Resolve());
  const std::vector<LotusIR::NodeIndex>* order;
  LOTUS_RETURN_IF_ERROR(graph.GetNodesInTopologicalOrder(&order));
  assert(order);
  TypeProto float_16_tensor_proto, float_tensor_proto;
  float_16_tensor_proto.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT16);
  float_tensor_proto.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  IdGenerator id_generator;
  std::map<LotusIR::NodeArg*, LotusIR::NodeArg*> input_def_updates;
  for (LotusIR::NodeIndex i : *order) {
    auto node = graph.GetNode(i);
    if (!node)
      return Status(LOTUS, INVALID_ARGUMENT);
    if (graph.IsSinkNode(*node) || graph.IsSourceNode(*node))
      continue;

    auto& inputs = node->InputDefs();
    std::map<const LotusIR::NodeArg*, LotusIR::NodeArg*> replacement_defs;
    bool casted = false;
    for (auto input : inputs) {
      if (NeedInsertCast(node, input)) {
        auto src_arg = const_cast<LotusIR::NodeArg*>(input);
        if (input_def_updates.count(src_arg)) {
          replacement_defs[src_arg] = input_def_updates[src_arg];
        } else {
          //insert cast op to cast input
          auto dst_arg = AddCastNode(graph,
                                     id_generator,
                                     src_arg,
                                     &float_tensor_proto,
                                     false,
                                     static_cast<int64_t>(TensorProto_DataType_FLOAT),
                                     //right now we only cast for cpu cases.
                                     LotusIR::kCpuExecutionProvider);
          replacement_defs[src_arg] = dst_arg;
          input_def_updates[src_arg] = dst_arg;
        }
        casted = true;
      }
    }

    if (casted && node->GetExecutionProviderType().empty()) {
      //set current node to CPU execution provider
      node->SetExecutionProviderType(kCpuExecutionProvider);
    }

    auto& outputs = node->OutputDefs();
    for (auto output : outputs) {
      // todo: check is the kernel available
      // here is based on the assumption that if we cast a cpu op's input from float16 to float
      // then this cpu op's output will become float.
      // not sure is it always correct...
      if (output->Type() &&
          DataTypeImpl::TypeFromProto(*output->TypeAsProto()) == DataTypeImpl::GetTensorType<MLFloat16>() &&
          casted) {
        //insert cast op to cast output back to float16
        auto dst_arg = const_cast<LotusIR::NodeArg*>(output);
        auto src_arg = AddCastNode(graph,
                                   id_generator,
                                   dst_arg,
                                   &float_tensor_proto,
                                   true,
                                   static_cast<int64_t>(TensorProto_DataType_FLOAT16),
                                   LotusIR::kCpuExecutionProvider);
        replacement_defs[dst_arg] = src_arg;
      }
    }

    node->ReplaceDefs(replacement_defs);
    modified |= casted;
  }
  //Resolve it to build the edges.
  LOTUS_RETURN_IF_ERROR(graph.Resolve());
  std::map<const LotusIR::NodeArg*, LotusIR::NodeArg*> replacement_defs;
  std::vector<LotusIR::NodeIndex> removed_nodes;
  for (auto& node : graph.Nodes()) {
    if (graph.IsSinkNode(node) || graph.IsSourceNode(node))
      continue;
    if (node.OpType() == "Cast") {
      // if cast's next node is also cast and next cast's output type equal to cast's input type
      // remove those two cast.
      auto src_type = node.InputDefs()[0]->Type();
      auto dst_type = node.OutputDefs()[0]->Type();
      auto input = node.InputDefs()[0];
      int child_removed = 0;
      int num_child = 0;
      for (auto it = node.OutputNodesBegin(); it != node.OutputNodesEnd(); it++) {
        if ((*it)->OpType() == "Cast") {
          auto src_type1 = (*it)->InputDefs()[0]->Type();
          auto dst_type1 = (*it)->OutputDefs()[0]->Type();
          if (src_type == dst_type1 && src_type1 == dst_type) {
            //node *it's output's follower could be linked with node's input.
            replacement_defs.clear();
            replacement_defs[const_cast<LotusIR::NodeArg*>((*it)->OutputDefs()[0])] = const_cast<LotusIR::NodeArg*>(input);
            for (auto next_it = (*it)->OutputNodesBegin(); next_it != (*it)->OutputNodesEnd(); next_it++) {
              const_cast<LotusIR::Node*>((*next_it))->ReplaceDefs(replacement_defs);
            }
            removed_nodes.push_back((*it)->Index());
            child_removed++;
          }
        }
        num_child++;
      }
      if (child_removed == num_child)
        removed_nodes.push_back(node.Index());
    }
  }

  for (auto i : removed_nodes) {
    graph.RemoveNode(i);
  }

  modified |= !removed_nodes.empty();
  return Status::OK();
}
}  // namespace Lotus
