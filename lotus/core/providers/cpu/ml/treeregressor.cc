#include "core/providers/cpu/ml/treeregressor.h"

namespace Lotus {
namespace ML {

ONNX_CPU_OPERATOR_ML_KERNEL(
    TreeEnsembleRegressor,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).MayInplace(0, 0),
    TreeEnsembleRegressor<float>);

template <typename T>
TreeEnsembleRegressor<T>::TreeEnsembleRegressor(const OpKernelInfo& info) : OpKernel(info) {
  info.GetAttrs<int64_t>("nodes_treeids", nodes_treeids_);
  info.GetAttrs<int64_t>("nodes_nodeids", nodes_nodeids_);
  info.GetAttrs<int64_t>("nodes_featureids", nodes_featureids_);
  info.GetAttrs<float>("nodes_values", nodes_values_);
  info.GetAttrs<float>("nodes_hitrates", nodes_hitrates_);
  info.GetAttrs<int64_t>("nodes_truenodeids", nodes_truenodeids_);
  info.GetAttrs<int64_t>("nodes_falsenodeids", nodes_falsenodeids_);
  info.GetAttrs<int64_t>("nodes_missing_value_tracks_true", missing_tracks_true_);
  info.GetAttrs<int64_t>("target_treeids", target_treeids_);
  info.GetAttrs<int64_t>("target_nodeids", target_nodeids_);
  info.GetAttrs<int64_t>("target_ids", target_ids_);
  info.GetAttrs<float>("target_weights", target_weights_);
  LOTUS_ENFORCE(info.GetAttr<int64_t>("n_targets", &n_targets_).IsOK());
  info.GetAttrs<float>("base_values", base_values_);

  //update nodeids to start at 0
  LOTUS_ENFORCE(!nodes_treeids_.empty());
  int64_t current_tree_id = 1234567891L;
  std::vector<int64_t> tree_offsets;

  for (size_t i = 0; i < nodes_treeids_.size(); i++) {
    if (nodes_treeids_[i] != current_tree_id) {
      tree_offsets.push_back(nodes_nodeids_[i]);
      current_tree_id = nodes_treeids_[i];
    }
    int64_t offset = tree_offsets[tree_offsets.size() - 1];
    nodes_nodeids_[i] = nodes_nodeids_[i] - offset;
    if (nodes_falsenodeids_[i] >= 0) {
      nodes_falsenodeids_[i] = nodes_falsenodeids_[i] - offset;
    }
    if (nodes_truenodeids_[i] >= 0) {
      nodes_truenodeids_[i] = nodes_truenodeids_[i] - offset;
    }
  }
  for (size_t i = 0; i < target_nodeids_.size(); i++) {
    int64_t offset = tree_offsets[target_treeids_[i]];
    target_nodeids_[i] = target_nodeids_[i] - offset;
  }

  std::vector<std::string> modes;
  info.GetAttrs<std::string>("nodes_modes", modes);

  for (const auto& mode : modes) {
    nodes_modes_.push_back(Lotus::ML::MakeTreeNodeMode(mode));
  }

  size_t nodes_id_size = nodes_nodeids_.size();
  LOTUS_ENFORCE(target_nodeids_.size() == target_ids_.size());
  LOTUS_ENFORCE(target_nodeids_.size() == target_weights_.size());
  LOTUS_ENFORCE(nodes_id_size == nodes_featureids_.size());
  LOTUS_ENFORCE(nodes_id_size == nodes_values_.size());
  LOTUS_ENFORCE(nodes_id_size == nodes_modes_.size());
  LOTUS_ENFORCE(nodes_id_size == nodes_truenodeids_.size());
  LOTUS_ENFORCE(nodes_id_size == nodes_falsenodeids_.size());
  LOTUS_ENFORCE((nodes_id_size == nodes_hitrates_.size()) || (0 == nodes_hitrates_.size()));

  std::string tmp = "SUM";
  info.GetAttr<std::string>("aggregate_function", &tmp);
  aggregate_function_ = Lotus::ML::MakeAggregateFunction(tmp);

  tmp = "NONE";
  info.GetAttr<std::string>("post_transform", &tmp);
  transform_ = Lotus::ML::MakeTransform(tmp);

  max_tree_depth_ = 1000;
  offset_ = four_billion_;
  //leafnode data, these are the votes that leaves do
  for (size_t i = 0; i < target_nodeids_.size(); i++) {
    leafnode_data_.push_back(std::make_tuple(target_treeids_[i], target_nodeids_[i], target_ids_[i], target_weights_[i]));
  }
  std::sort(begin(leafnode_data_), end(leafnode_data_), [](auto const& t1, auto const& t2) {
    if (std::get<0>(t1) != std::get<0>(t2))
      return std::get<0>(t1) < std::get<0>(t2);
    else
      return std::get<1>(t1) < std::get<1>(t2);
  });
  //make an index so we can find the leafnode data quickly when evaluating
  int64_t field0 = -1;
  int64_t field1 = -1;
  for (size_t i = 0; i < leafnode_data_.size(); i++) {
    int64_t id0 = std::get<0>(leafnode_data_[i]);
    int64_t id1 = std::get<1>(leafnode_data_[i]);
    if (id0 != field0 || id1 != field1) {
      int64_t id = id0 * four_billion_ + id1;
      auto p3 = std::make_pair(id, i);  // position is i
      leafdata_map_.insert(p3);
      field0 = id;
      field1 = static_cast<int64_t>(i);
    }
  }
  //treenode ids, some are roots, and roots have no parents
  std::unordered_map<int64_t, size_t> parents;  //holds count of all who point to you
  std::unordered_map<int64_t, size_t> indices;
  //add all the nodes to a map, and the ones that have parents are not roots
  std::unordered_map<int64_t, size_t>::iterator it;
  size_t start_counter = 0L;
  for (size_t i = 0; i < nodes_treeids_.size(); i++) {
    //make an index to look up later
    int64_t id = nodes_treeids_[i] * four_billion_ + nodes_nodeids_[i];
    auto p3 = std::make_pair(id, i);  // i is the position
    indices.insert(p3);
    it = parents.find(id);
    if (it == parents.end()) {
      //start counter at 0
      auto p1 = std::make_pair(id, start_counter);
      parents.insert(p1);
    }
  }
  //all true nodes aren't roots
  for (size_t i = 0; i < nodes_truenodeids_.size(); i++) {
    if (nodes_modes_[i] == Lotus::ML::NODE_MODE::LEAF) continue;
    //they must be in the same tree
    int64_t id = nodes_treeids_[i] * offset_ + nodes_truenodeids_[i];
    it = parents.find(id);
    LOTUS_ENFORCE(it != parents.end());
    it->second++;
  }
  //all false nodes aren't roots
  for (size_t i = 0; i < nodes_falsenodeids_.size(); i++) {
    if (nodes_modes_[i] == Lotus::ML::NODE_MODE::LEAF) continue;
    //they must be in the same tree
    int64_t id = nodes_treeids_[i] * offset_ + nodes_falsenodeids_[i];
    it = parents.find(id);
    LOTUS_ENFORCE(it != parents.end());
    it->second++;
  }
  //find all the nodes that dont have other nodes pointing at them
  for (auto& parent : parents) {
    if (parent.second == 0) {
      int64_t id = parent.first;
      it = indices.find(id);
      roots_.push_back(it->second);
    }
  }
  LOTUS_ENFORCE(base_values_.empty() || base_values_.size() == static_cast<size_t>(n_targets_));
}

template <typename T>
Common::Status TreeEnsembleRegressor<T>::ProcessTreeNode(std::unordered_map<int64_t, float>& classes, int64_t treeindex, const T* Xdata, int64_t feature_base) const {
  //walk down tree to the leaf
  Lotus::ML::NODE_MODE mode = static_cast<Lotus::ML::NODE_MODE>(nodes_modes_[treeindex]);
  int64_t loopcount = 0;
  int64_t root = treeindex;
  while (mode != Lotus::ML::NODE_MODE::LEAF) {
    T val = Xdata[feature_base + nodes_featureids_[treeindex]];
    bool tracktrue = true;
    if (missing_tracks_true_.size() != nodes_truenodeids_.size()) {
      tracktrue = false;
    } else {
      tracktrue = (missing_tracks_true_[treeindex] != 0) && std::isnan(val);
    }
    float threshold = nodes_values_[treeindex];
    if (mode == Lotus::ML::NODE_MODE::BRANCH_LEQ) {
      treeindex = val <= threshold || tracktrue ? nodes_truenodeids_[treeindex] : nodes_falsenodeids_[treeindex];
    } else if (mode == Lotus::ML::NODE_MODE::BRANCH_LT) {
      treeindex = val < threshold || tracktrue ? nodes_truenodeids_[treeindex] : nodes_falsenodeids_[treeindex];
    } else if (mode == Lotus::ML::NODE_MODE::BRANCH_GTE) {
      treeindex = val >= threshold || tracktrue ? nodes_truenodeids_[treeindex] : nodes_falsenodeids_[treeindex];
    } else if (mode == Lotus::ML::NODE_MODE::BRANCH_GT) {
      treeindex = val > threshold || tracktrue ? nodes_truenodeids_[treeindex] : nodes_falsenodeids_[treeindex];
    } else if (mode == Lotus::ML::NODE_MODE::BRANCH_EQ) {
      treeindex = val == threshold || tracktrue ? nodes_truenodeids_[treeindex] : nodes_falsenodeids_[treeindex];
    } else if (mode == Lotus::ML::NODE_MODE::BRANCH_NEQ) {
      treeindex = val != threshold || tracktrue ? nodes_truenodeids_[treeindex] : nodes_falsenodeids_[treeindex];
    }

    if (treeindex < 0) {
      return Common::Status(Common::LOTUS, Common::RUNTIME_EXCEPTION,
                            "treeindex evaluated to a negative value, which should not happen.");
    }
    treeindex = treeindex + root;
    mode = (Lotus::ML::NODE_MODE)nodes_modes_[treeindex];
    loopcount++;
    if (loopcount > max_tree_depth_) break;
  }
  //should be at leaf
  int64_t id = nodes_treeids_[treeindex] * four_billion_ + nodes_nodeids_[treeindex];
  //auto it_lp = leafdata_map.find(id);
  auto it_lp = leafdata_map_.find(id);
  if (it_lp != leafdata_map_.end()) {
    size_t index = it_lp->second;
    int64_t treeid = std::get<0>(leafnode_data_[index]);
    int64_t nodeid = std::get<1>(leafnode_data_[index]);
    while (treeid == nodes_treeids_[treeindex] && nodeid == nodes_nodeids_[treeindex]) {
      int64_t classid = std::get<2>(leafnode_data_[index]);
      float weight = std::get<3>(leafnode_data_[index]);
      auto it_classes = classes.find(classid);
      if (it_classes != classes.end()) {
        it_classes->second += weight;
      } else {
        auto p1 = std::make_pair(classid, weight);
        classes.insert(p1);
      }
      index++;
      if (index >= leafnode_data_.size()) {
        break;
      }
      treeid = std::get<0>(leafnode_data_[index]);
      nodeid = std::get<1>(leafnode_data_[index]);
    }
  }
  return Common::Status::OK();
}

template <typename T>
Common::Status TreeEnsembleRegressor<T>::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  if (X->Shape().Size() == 0) {
    return Status(Common::LOTUS, Common::INVALID_ARGUMENT,
                  "Input shape needs to be at least a single dimension.");
  }

  int64_t stride = X->Shape().Size() == 1 ? X->Shape()[0] : X->Shape()[1];
  int64_t N = X->Shape().Size() == 1 ? 1 : X->Shape()[0];
  Tensor* Y = context->Output(0, TensorShape({N, n_targets_}));

  int64_t write_index = 0;
  const auto* x_data = X->Data<T>();

  for (int64_t i = 0; i < N; i++)  //for each class
  {
    int64_t current_weight_0 = i * stride;
    std::unordered_map<int64_t, float> scores;
    //for each tree
    for (size_t j = 0; j < roots_.size(); j++) {
      //walk each tree from its root
      LOTUS_RETURN_IF_ERROR(ProcessTreeNode(scores, roots_[j], x_data, current_weight_0));
    }
    //find aggregate, could use a heap here if there are many classes
    std::vector<float> outputs;
    for (int64_t j = 0; j < n_targets_; j++) {
      //reweight scores based on number of voters
      auto it_scores = scores.find(j);
      float val = base_values_.size() == (size_t)n_targets_ ? base_values_[j] : 0.f;
      if (it_scores != scores.end()) {
        if (aggregate_function_ == Lotus::ML::AGGREGATE_FUNCTION::AVERAGE) {
          val += scores[j] / roots_.size();
        } else if (aggregate_function_ == Lotus::ML::AGGREGATE_FUNCTION::SUM) {
          val += scores[j];
        } else if (aggregate_function_ == Lotus::ML::AGGREGATE_FUNCTION::MIN) {
          if (scores[j] < val) val = scores[j];
        } else if (aggregate_function_ == Lotus::ML::AGGREGATE_FUNCTION::MAX) {
          if (scores[j] > val) val = scores[j];
        }
      }
      outputs.push_back(val);
    }
    if (transform_ == Lotus::ML::POST_EVAL_TRANSFORM::LOGISTIC) {
      for (float& output : outputs) {
        output = Lotus::ML::ml_logit(output);
      }
    } else if (transform_ == Lotus::ML::POST_EVAL_TRANSFORM::SOFTMAX) {
      Lotus::ML::compute_softmax(outputs);
    } else if (transform_ == Lotus::ML::POST_EVAL_TRANSFORM::SOFTMAX_ZERO) {
      Lotus::ML::compute_softmax_zero(outputs);
    }
    for (float output : outputs) {
      Y->MutableData<float>()[write_index] = output;
      write_index++;
    }
  }
  return Status::OK();
}

}  // namespace ML
}  // namespace Lotus
