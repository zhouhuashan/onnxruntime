#include "core/providers/brainslice/gru.h"
#include "core/providers/brainslice/brain_slice_execution_provider.h"
#include "Lstm_client.h"
namespace onnxruntime {
namespace brainslice{
ONNX_OPERATOR_KERNEL_EX(
    GRU,
    kOnnxDomain,
    7,
	kBrainSliceExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).MemoryType<ONNXRuntimeMemTypeCPUOutput>(0).MemoryType<ONNXRuntimeMemTypeCPUOutput>(1).InputMemoryType<ONNXRuntimeMemTypeCPUInput>(0).InputMemoryType<ONNXRuntimeMemTypeCPUInput>(1).InputMemoryType<ONNXRuntimeMemTypeCPUInput>(2).InputMemoryType<ONNXRuntimeMemTypeCPUInput>(3),
    BrainSliceGRU<float>);

//TODO: move to util
static void Convert2Float16AndPad(const std::vector<std::vector<float>>& src,
                                  std::vector<std::vector<BS_Half>>& dst,
                                  size_t num_rows, size_t num_cols) {
  for (size_t i = 0; i < src.size(); i++) {
    for (size_t j = 0; j < src[i].size(); j++) {
      dst[i].push_back(BS_Half(src[i][j]));
    }
    dst[i].insert(dst[i].end(), num_cols - src[i].size(), BS_Half());
  }

  std::vector<BS_Half> zero(num_cols, BS_Half());
  for (size_t k = src.size(); k < num_rows; k++) {
    dst.insert(dst.end(), num_rows - src.size(), zero);
  }
}

template <>
BrainSliceGRU<float>::BrainSliceGRU(const OpKernelInfo& info) : BrainSliceOpKernel(info) {
  std::string direction;
  ONNXRUNTIME_ENFORCE(info.GetAttr("direction", &direction).IsOK());
  if (std::strcmp(direction.c_str(), "forward") != 0) {
    ONNXRUNTIME_NOT_IMPLEMENTED("GRU with %s direction is not supported.", direction);
  }
  ONNXRUNTIME_ENFORCE(info.GetAttr("hidden_size", &hidden_).IsOK());
  auto& capabilities = provider_->GetFPGAHandle().GetCapacities();
  //1. convert weights to BrainSliceParameterInitPlan
  std::vector<BrainSliceParameterInitPlan> parameters;
  //a. W
  const Tensor* W;
  ONNXRUNTIME_ENFORCE(info.TryGetConstantInput(1, &W), "GRU's W must be a initializers.");
  auto w_dims = W->Shape().GetDims();
  assert(w_dims.size() == 3 && w_dims[0] == 1 && (w_dims[1] % 3) == 0);
  TensorShape w_shape({w_dims[0], w_dims[1] / 3, w_dims[2]});
  char* w_buffer = static_cast<char*>(const_cast<void*>((W->DataRaw())));

  BrainSliceParameterInitPlan wr_plan = {nullptr, ParameterUsage::USE_AS_MATRIX, 2, false, ISA_Mem_MatrixRf, 0};
  wr_plan.tensor = std::make_unique<Tensor>(W->DataType(), w_shape, w_buffer + w_shape.Size() * W->DataType()->Size(), W->Location());
  BrainSliceParameterInitPlan wz_plan = {nullptr, ParameterUsage::USE_AS_MATRIX, 2, false, ISA_Mem_MatrixRf, 0};
  wz_plan.tensor = std::make_unique<Tensor>(W->DataType(), w_shape, w_buffer, W->Location());
  BrainSliceParameterInitPlan wh_plan = {nullptr, ParameterUsage::USE_AS_MATRIX, 2, false, ISA_Mem_MatrixRf, 0};
  wh_plan.tensor = std::make_unique<Tensor>(W->DataType(), w_shape, w_buffer + 2 * (w_shape.Size() * W->DataType()->Size()), W->Location());

  //b. R
  const Tensor* R;
  ONNXRUNTIME_ENFORCE(info.TryGetConstantInput(2, &R), "GRU's R must be a initializers.");
  auto r_dims = R->Shape().GetDims();
  assert(r_dims.size() == 3 && r_dims[0] == 1 && (r_dims[1] % 3) == 0);
  ONNXRUNTIME_ENFORCE(hidden_ == r_dims[2]);
  TensorShape r_shape({r_dims[0], r_dims[1] / 3, r_dims[2]});
  char* r_buffer = static_cast<char*>(const_cast<void*>((R->DataRaw())));
  BrainSliceParameterInitPlan rr_plan = {nullptr, ParameterUsage::USE_AS_MATRIX, 2, false, ISA_Mem_MatrixRf, 0};
  rr_plan.tensor = std::make_unique<Tensor>(R->DataType(), r_shape, r_buffer + r_shape.Size() * R->DataType()->Size(), R->Location());
  BrainSliceParameterInitPlan rz_plan = {nullptr, ParameterUsage::USE_AS_MATRIX, 2, false, ISA_Mem_MatrixRf, 0};
  rz_plan.tensor = std::make_unique<Tensor>(R->DataType(), r_shape, r_buffer, R->Location());
  BrainSliceParameterInitPlan rh_plan = {nullptr, ParameterUsage::USE_AS_MATRIX, 2, false, ISA_Mem_MatrixRf, 0};
  rh_plan.tensor = std::make_unique<Tensor>(R->DataType(), r_shape, r_buffer + 2 * (r_shape.Size() * R->DataType()->Size()), R->Location());

  //The built-in GRU firmware is trick that it assume the matrix are load in the order: Wr, Rr, Wz, Rz, Wh, Rh
  //And need to start from address 0. So the order matters here
  parameters.push_back(std::move(wr_plan));
  parameters.push_back(std::move(rr_plan));
  parameters.push_back(std::move(wz_plan));
  parameters.push_back(std::move(rz_plan));
  parameters.push_back(std::move(wh_plan));
  parameters.push_back(std::move(rh_plan));

  //2. upload the weights
  for (auto& parameter : parameters) {
    ONNXRUNTIME_ENFORCE(BrainSliceOpKernel::UploadBrainSliceParameter<float>(parameter, provider_).IsOK());
  }
  //3. call GRU init function
  std::vector<std::vector<BS_Half>> half_bias(3);
  const Tensor* B;
  if (info.TryGetConstantInput(3, &B)) {
    auto b_dims = B->Shape().GetDims();
    assert(b_dims.size() == 2 && b_dims[0] == 1 && b_dims[1] % 6 == 0);
    const float* data = B->Data<float>();
    //TODO: float only now
    std::vector<std::vector<float>> bias;
    //Wbz:
    std::vector<float> bz(hidden_);
    bz.assign(data, data + hidden_);
    for (int i = 0; i < hidden_; ++i) {
      bz[i] += *(data + 3 * hidden_ + i);
    }
    //Wbr:
    std::vector<float> br(hidden_);
    br.assign(data + hidden_, data + 2 * hidden_);
    for (int i = 0; i < hidden_; ++i) {
      br[i] += *(data + 4 * hidden_ + i);
    }
    //Wbh:
    std::vector<float> bh(hidden_);
    bh.assign(data + 2 * hidden_, data + 3 * hidden_);
    for (int i = 0; i < hidden_; ++i) {
      bh[i] += *(data + 5 * hidden_ + i);
    }

    bias.push_back(std::move(br));
    bias.push_back(std::move(bz));
    bias.push_back(std::move(bh));

    Convert2Float16AndPad(bias, half_bias, bias.size(), ((hidden_ + native_dim_ - 1) / native_dim_) * native_dim_);
  } else {
    half_bias[0].resize(hidden_, BS_Half(0.0f));
    half_bias[1].resize(hidden_, BS_Half(0.0f));
    half_bias[2].resize(hidden_, BS_Half(0.0f));
  }
  //TODO: input_dim need to be know statically
  auto& inputs = info.node().InputDefs();
  assert(inputs.size() > 0);
  auto input_shape = inputs[0]->Shape();
  ONNXRUNTIME_ENFORCE(input_shape, "GRU require input has static shape");
  auto& shape = input_shape->dim();
  assert(shape.size() == 3);
  ONNXRUNTIME_ENFORCE(shape[2].dim_param().empty(), "GRU's input dimension need to be static.");
  auto input_dim = static_cast<size_t>(shape[2].dim_value());
  // Initialize GRU and Load Bias
  Lstm_InitGruParams init_args;
  init_args.inputDim = static_cast<uint32_t>(((input_dim + native_dim_ - 1) / native_dim_) * native_dim_);
  init_args.outputDim = static_cast<uint32_t>(((hidden_ + native_dim_ - 1) / native_dim_) * native_dim_);
  ONNXRUNTIME_ENFORCE(provider_->GetFPGAHandle().SendSync(
                                                    [&](void* request, size_t* request_size) {
                                                      void* zero = _alloca(init_args.outputDim * sizeof(BS_Half));
                                                      memset(zero, 0, init_args.outputDim * sizeof(BS_Half));
                                                      const void* bias_addr[3];
                                                      std::transform(half_bias.begin(), half_bias.end(), bias_addr, [](auto& v) { return v.data(); });

                                                      return Lstm_Functions_InitGruAndLoadBias_Request_Float16(
                                                          &capabilities.m_bsParameters,
                                                          &init_args,
                                                          zero,
                                                          bias_addr,
                                                          request, request_size);
                                                    },
                                                    [&](const void* response, size_t response_size) {
                                                      return Lstm_Functions_InitGruAndLoadBias_Response(
                                                          &capabilities.m_bsParameters,
                                                          response, response_size);
                                                    })
                          .IsOK());
}

template <>
Status BrainSliceGRU<float>::Compute(OpKernelContext* context) const {
  auto X = context->Input<Tensor>(0);
  auto& input_shape = X->Shape();
  auto batch_size = input_shape[1];
  if (batch_size != 1)
    return Status(common::ONNXRUNTIME, common::FAIL, "BrainSlice GRU only support batch size 1.");
  //1. prepare the input
  auto* data = X->Data<float>();
  auto seq_len = input_shape[0];
  auto input_dim = input_shape[2];
  std::vector<std::vector<BS_Half>> half_inputs;
  auto step_dim = ((input_dim + native_dim_ - 1) / native_dim_) * native_dim_;
  for (auto i = 0; i < seq_len; ++i) {
    std::vector<BS_Half> half_data;
    for (auto j = 0; j < input_dim; ++j) {
      half_data.push_back(BS_Half(*(data + input_dim * i + j)));
    }
    half_data.resize(step_dim);
    half_inputs.push_back(half_data);
  }

  auto Y = context->Output(0, TensorShape({seq_len, 1, batch_size, hidden_}));
  auto Y_h = context->Output(1, TensorShape({1, batch_size, hidden_}));
  if (!Y && !Y_h)  // nothing need to be calculated.
    return Status::OK();

  Lstm_EvalGruParams eval_args;
  eval_args.rnnSteps = static_cast<uint32_t>(seq_len);
  eval_args.inputDim = static_cast<uint32_t>(step_dim);
  eval_args.outputDim = static_cast<uint32_t>(((hidden_ + native_dim_ - 1) / native_dim_) * native_dim_);
  eval_args.exportHidden = Y == nullptr ? 0 : 1;

  auto& capabilities = provider_->GetFPGAHandle().GetCapacities();

  return provider_->GetFPGAHandle().SendSync(
      [&](void* request, size_t* request_size) {
        auto addr_X = static_cast<const void**>(_alloca(eval_args.rnnSteps * sizeof(const void*)));
        std::transform(half_inputs.begin(), half_inputs.end(), addr_X, [](auto& v) { return v.data(); });

        auto status = Lstm_Functions_EvaluateGru_Request_Float16(
            &capabilities.m_bsParameters,
            &eval_args,
            addr_X,
            request, request_size);

        return status;
      },
      [&](const void* response, size_t response_size) {
        size_t output_size, output_count = (eval_args.exportHidden ? eval_args.rnnSteps : 1);
        auto addr_Y = static_cast<const void**>(_alloca(output_count * sizeof(const void*)));

        auto status = Lstm_Functions_EvaluateGru_Response_Float16(
            &capabilities.m_bsParameters,
            &eval_args,
            response, response_size,
            addr_Y, &output_size, &output_count);

        if (eval_args.exportHidden) {
          auto* y_data = Y->MutableData<float>();
          assert(output_count == static_cast<size_t>(seq_len) && output_size == eval_args.outputDim);
          for (auto i = 0; i < output_count; ++i) {
            for (auto j = 0; j < hidden_; ++j) {
              *(y_data + i * hidden_ + j) = *(static_cast<const BS_Half*>(addr_Y[i]) + j);
            }
          }
          if (Y_h) {
            auto* y_h_data = Y_h->MutableData<float>();
            memcpy(y_h_data, y_data + seq_len * hidden_ - hidden_, sizeof(float) * hidden_);
          }
        } else {
          auto y_h_data = Y_h->MutableData<float>();
          assert(output_count == 1 && output_size == eval_args.outputDim);
          for (auto i = 0; i < hidden_; ++i) {
            *(y_h_data + i) = *(static_cast<const BS_Half*>(*addr_Y) + i);
          }
        }

        return status;
      });
}
}
}  // namespace onnxruntime
