#pragma once

#include "core/common/status.h"
#include "core/common/logging/logging.h"
#include "core/framework/op_kernel.h"
#include "core/graph/schema_registry.h"

namespace Lotus {
// TODO - Move KernelCreateFn somewhere common to avoid the duplicate function.
using CustomKernelCreateFn = std::function<OpKernel*(const OpKernelInfo& info)>;

class CustomRegistry : public KernelRegistry, public LotusIR::LotusOpSchemaRegistry {
 public:
  CustomRegistry(bool create_func_kernel);

  ~CustomRegistry() = default;

  /**
    * Register a kernel definition together with kernel factory method to this session.
    * If any conflict happened between registered kernel def and built-in kernel def,
    * registered kernel will have higher priority.
    * Call this before invoking Initialize().
    * @return OK if success.
    */
  Common::Status RegisterCustomKernel(KernelDefBuilder& kernel_def_builder, CustomKernelCreateFn kernel_creator);

  /**
  * Register a onnx opset to this session.
  * If any conflict happened between registered schema and built-in schema,
  * registered schema will have higher priority.
  * Call this before invoking Load().
  * @return OK if success.
  */
  Common::Status RegisterCustomOpSet(std::vector<OpSchema>& schemas, const std::string& domain, int version);

 private:
  CustomRegistry() = delete;

  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(CustomRegistry);
};

class CustomRegistryManager : public LotusIR::ILotusOpSchemaCollection {
 public:
  void RegisterCustomRegistry(std::shared_ptr<CustomRegistry> custom_registry);

  Status CreateKernel(const LotusIR::Node& node,
                      const IExecutionProvider* execution_provider,
                      /*out*/ std::unique_ptr<OpKernel>* op_kernel) const;

  Status SearchKernelRegistry(const LotusIR::Node& node,
                              /*out*/ const KernelRegistry::KernelCreateInfo** kernel_create_info) const;

  bool HasSchema() const;

  LotusIR::Domain_To_Version_Map DomainToVersionMap() const override;

  const ONNX_NAMESPACE::OpSchema* Schema(
      const std::string& key,
      const std::string& domain = LotusIR::kOnnxDomain) const override;

  const ONNX_NAMESPACE::OpSchema* Schema(
      const std::string& key,
      const int maxInclusiveVersion,
      const std::string& domain = LotusIR::kOnnxDomain) const override;

  std::vector<const KernelRegistry*> GetAllKernelRegistries() {
    std::vector<const KernelRegistry*> result;
    for (auto& registry : custom_registries) {
      result.push_back(registry.get());
    }
    return result;
  }

 private:
  std::list<std::shared_ptr<CustomRegistry>> custom_registries;
};
}  // namespace Lotus
