#include <unordered_map>
#include <string>
#include <cstdint>
#include <memory>
#include <functional>

namespace onnx {
class ValueInfoProto;
class TensorProto;
class TypeProto;
class AttributeProto;
}  // namespace onnx

namespace LotusIR {
using NodeIndex = size_t;
using Version = int64_t;
using NodeArgInfo = onnx::ValueInfoProto;
using InitializedTensorSet = std::unordered_map<std::string, const onnx::TensorProto*>;
using ArgNameToTypeMap = std::unordered_map<std::string, onnx::TypeProto>;
using ProviderType = const std::string&;
// TODO - Evaluate switching the types below to support transparent comparators and enable
// lookups based on gsl::cstring_span<> and std::string_view.  This would reduces allocations
// converting to std::string, but requires conversion to std::map<std::string, foo, std::less<>>
// instead of std::unordered_map<std::string, foo, [std::less<foo>]>.

using NodeAttributes = std::unordered_map<std::string, onnx::AttributeProto>;
class ILotusOpSchemaCollection;
using ILotusOpSchemaCollectionPtr = std::shared_ptr<ILotusOpSchemaCollection>;
}  // namespace LotusIR

namespace Lotus {
class OpKernel;
class OpKernelInfo;

using KernelCreateFn = std::function<OpKernel*(const OpKernelInfo& info)>;
}  // namespace Lotus
