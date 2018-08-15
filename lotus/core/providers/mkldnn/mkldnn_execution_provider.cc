#include "mkldnn_allocator.h"
#include "mkldnn_execution_provider.h"
#include "core/framework/allocator.h"
#include "core/framework/transformer_memcpy.h"
#include "core/framework/memcpy.h"

namespace Lotus {
namespace MklDnn {

ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kMklDnnExecutionProvider,
    KernelDefBuilder().InputMemoryType<kMemTypeCPUInput>(0).TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kMklDnnExecutionProvider,
    KernelDefBuilder().OutputMemoryType<kMemTypeCPUOutput>(0).TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Memcpy);

}  // namespace MklDnn

MKLDNNExecutionProvider::MKLDNNExecutionProvider(const MKLDNNExecutionProviderInfo& /*info*/) {
  DeviceAllocatorRegistrationInfo default_allocator_info({kMemTypeDefault,
                                                          [](int) { return std::make_unique<MKLDNNAllocator>(); }, std::numeric_limits<size_t>::max()});
  InsertAllocator(kMemTypeDefault, CreateAllocator(default_allocator_info));

  DeviceAllocatorRegistrationInfo cpu_allocator_info({kMemTypeCPUOutput,
                                                      [](int) { return std::make_unique<MKLDNNCPUAllocator>(); }, std::numeric_limits<size_t>::max()});
  InsertAllocator(kMemTypeCPUOutput, CreateAllocator(cpu_allocator_info));
}

MKLDNNExecutionProvider::~MKLDNNExecutionProvider() {
}

Status MKLDNNExecutionProvider::CopyTensor(const Tensor& src, Tensor& dst) const {
  // Support CPU <-> MKLDNN for now
  if (!(src.Location().name == MKLDNN && dst.Location().name == CPU) &&
      !(src.Location().name == CPU && dst.Location().name == MKLDNN) &&
      !(src.Location().name == MKLDNN && dst.Location().name == MKLDNN_CPU)) {
    LOTUS_NOT_IMPLEMENTED(src.Location().name + " copy to ", dst.Location().name, " is not implemented");
  }

  // Todo: Copy for now. May optimize later to avoid copy.
  size_t bytes = src.DataType()->Size() * src.Shape().Size();
  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();
  memcpy(dst_data, src_data, bytes);

  return Status::OK();
}

namespace MklDnn {
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, Conv);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 7, Gemm);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, MemcpyFromHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, MemcpyToHost);
void RegisterMKLDNNKernels(std::function<void(KernelCreateInfo&&)> fn) {
  fn(BuildKernel<ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, Conv)>());
  fn(BuildKernel<ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 7, Gemm)>());
  fn(BuildKernel<ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>());
  fn(BuildKernel<ONNX_OPERATOR_KERNEL_CLASS_NAME(kMklDnnExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>());
}
}  // namespace MklDnn

std::shared_ptr<KernelRegistry> MKLDNNExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>(Lotus::MklDnn::RegisterMKLDNNKernels);
  return kernel_registry;
}
}  // namespace Lotus
