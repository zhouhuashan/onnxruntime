#include <tvm/runtime/device_api.h>

#include "nuphar_execution_provider.h"
#include "core/codegen_utils/tvm_utils.h"

namespace onnxruntime {

namespace nuphar {

static void RegisterNupharKernels(std::function<void(KernelCreateInfo&&)> fn) {
}

}  // namespace nuphar

NupharExecutionProvider::~NupharExecutionProvider() {
}

Status NupharExecutionProvider::CopyTensor(const Tensor& src, Tensor& dst) const {
  if (!(src.Location().name == TVM_STACKVM && dst.Location().name == TVM_STACKVM))
    LOTUS_NOT_IMPLEMENTED("copy to ", dst.Location().name, " from ", src.Location().name, " is not implemented");

  size_t bytes = src.DataType()->Size() * src.Shape().Size();
  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();

  tvm::runtime::DeviceAPI::Get(tvm_ctx_)->CopyDataFromTo(
    src_data, /*src_byte_offset*/0, dst_data, /*dst_byte_offset*/0, bytes, /*src_ctx*/tvm_ctx_,
    /*dst_ctx*/tvm_ctx_, /*data_type*/tvm_codegen::ToTvmDLDataType(src.DataType()), /*stream*/nullptr);

  return Status::OK();
}

std::shared_ptr<KernelRegistry> NupharExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>(onnxruntime::nuphar::RegisterNupharKernels);
  return kernel_registry;
}

}  // namespace onnxruntime
