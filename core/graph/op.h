#ifndef CORE_GRAPH_OP_H
#define CORE_GRAPH_OP_H

#include <functional>
#include <unordered_map>

#include "opschema.h"
#include "shape_inference.h"

namespace LotusIR
{
    class OperatorSchema;
#ifdef ONNX_V1_OPSCHEMA_COMPAT
    class OperatorDefinitionSetter;
    typedef OperatorDefinitionSetter OpSchema;
#endif // #ifdef ONNX_V1_OPSCHEMA_COMPAT

    class TypeUtils
    {
    public:

        // Get attribute type given attribute proto data.
        static Status GetType(const AttributeProto& p_attr, AttrType& p_type);

    };

    // An attribute parser - it's specified when registering an operator.
    // The parser is designed and used in two ways.
    // 1) It will be used to verify whether a Node's attributes match the
    //    operator's definition.
    // 2) It will be used to parse a Node's attributes into a <T> object,
    //    which makes it be easier to access node attributes.
    // TODO: to implement the 2nd point above, NodeAttributes should be changed
    // to contain a <T> field, which is structured attributes.
    typedef std::function<Status(const NodeAttributes&)> AttributeParser;

    class OperatorDefinition
    {
    public:

        const std::string& GetName() const;
        const OperatorSchema& GetOpSchema() const;
        ShapeInferenceFunc GetShapeInferenceFn() const;
        AttributeParser GetAttributeParser() const;

    private:

        friend class OperatorDefinitionSetter;
        friend class OperatorDefinitionRegistry;

        OperatorSchema m_opSchema;
        ShapeInferenceFunc m_shapeInferenceFunc;
        AttributeParser m_attrParser;
    };

    typedef std::tuple<std::string, std::string, std::string> InputOutputParam;
    typedef std::tuple<std::string, std::string, AttrType, AttributeProto> AttrParam;
    typedef std::tuple<std::string, std::vector<std::string>, std::string> TypeConstraintParam;

#define ATTR_SETTER_INTERFACE(TypeName) \
    OperatorDefinitionSetter& Attr(const std::string& p_attrName, \
                               const std::string& p_description, \
                               AttrType p_attrType, \
                               const TypeName& p_defaultValue); \
    OperatorDefinitionSetter& Attr(const std::string& p_attrName, \
                               const std::string& p_description, \
                               AttrType p_attrType, \
                               const std::vector<TypeName>& p_defaultValues); \

    // Operator registry setter helper.
    // This is used in "OPERATOR_DEFINITION" macro, to separate setters from getters
    // in OperatorSchema.
    class OperatorDefinitionSetter
    {
    public:

        OperatorDefinitionSetter() = default;

        OperatorDefinitionSetter& Name(const std::string& p_opName);

        OperatorDefinitionSetter& Description(const std::string& p_description);

        OperatorDefinitionSetter& Input(const std::string& p_inputName,
            const std::string& p_description,
            const std::string& p_type = "");

        OperatorDefinitionSetter& Output(const std::string& p_outputName,
            const std::string& p_description,
            const std::string& p_type = "");

        OperatorDefinitionSetter& Attr(const std::string& p_attrName,
            const std::string& p_description,
            AttrType p_attrType, bool required = false);

        ATTR_SETTER_INTERFACE(int64_t)
        ATTR_SETTER_INTERFACE(float)
        ATTR_SETTER_INTERFACE(std::string)
        ATTR_SETTER_INTERFACE(TensorProto)
        ATTR_SETTER_INTERFACE(GraphProto)
        ATTR_SETTER_INTERFACE(TypeProto)
        ATTR_SETTER_INTERFACE(TensorShapeProto)

        OperatorDefinitionSetter& TypeConstraint(const std::string& p_typeName,
            const std::vector<std::string>& p_constraints,
            const std::string& p_description);

        // Shape inference function will be used to infer outputs' shape with
        // inputs' shape.
        OperatorDefinitionSetter& SetShapeInferenceFunc(
            ShapeInferenceFunc p_shapeInferFunc);

        // Attribute parser will be used to parse Node's attributes to see
        // whether Node attributes are matching operator attributes definition.
        OperatorDefinitionSetter& SetAttributeParser(
            AttributeParser p_attrParser);

#ifdef ONNX_V1_OPSCHEMA_COMPAT
        enum class SupportType {
            COMMON,
            EXPERIMENTAL,
        };
        // Methods added for compatibility with ONNX OpSchema registration API
        OpSchema& NumInputs(int n)
        {
            return NumInputs(n, n);
        }
        OpSchema& NumInputs(int min, int max)
        {
            m_opDefData.m_opSchema.m_onnxMinInput = min;
            m_opDefData.m_opSchema.m_onnxMaxInput = max;
            return *this;
        }
        OpSchema& NumInputs(std::set<int> allowed_input_nums)
        {
            return NumInputs([allowed_input_nums](int n)-> bool {
                return allowed_input_nums.count(n) > 0;
            });
        }
        OpSchema& NumInputs(std::function<bool(int)> func)
        {
            m_opDefData.m_opSchema.m_onnxNumInputsAllowed = func;
            return *this;
        }
        OpSchema& NumOutputs(int n) {
            return NumOutputs(n, n);
        }
        OpSchema& NumOutputs(int min, int max)
        {
            m_opDefData.m_opSchema.m_onnxMinOutput = min;
            m_opDefData.m_opSchema.m_onnxMaxOutput = max;
            return *this;
        }
        OpSchema& NumOutputs(std::set<int> allowed_output_nums)
        {
            return NumOutputs([allowed_output_nums](int n)-> bool {
                return allowed_output_nums.count(n) > 0;
            });
        }
        OpSchema& NumOutputs(std::function<bool(int)> func)
        {
            m_opDefData.m_opSchema.m_onnxNumOutputsAllowed = func;
            return *this;
        }
        OpSchema& NumInputsOutputs(std::function<bool(int, int)> func)
        {
            m_opDefData.m_opSchema.m_onnxNumInputsOutputsAllowed = func;
            return *this;
        }
        OpSchema& OutputCalculator(std::function<int(int)> calc) { return *this; }
        OpSchema& SameNumberOfOutput() { return *this; }
        OpSchema& AllowConsumed(std::function<std::pair<bool, int>(int)> inplace) { return *this; }
        OpSchema& AllowConsumed(std::unordered_map<int, int> inplace) { return *this; }
        OpSchema& AllowOneToOneConsumed() { return *this; }
        OpSchema& EnforceConsumed(std::function<std::pair<bool, int>(int)> inplace) { return *this; }
        OpSchema& EnforceConsumed(std::unordered_map<int, int> inplace) { return *this; }
        OpSchema& EnforceOneToOneConsumed() { return *this; }
        OpSchema& SetSupportLevel(SupportType supportType) { return *this; }
        OpSchema& AllowUncheckedAttributes() { return *this; }
        OpSchema& FillUsing(std::function<void(OpSchema&)> populator)
        {
            if (populator)
            {
                populator(*this);
            }
            return *this;
        }
        OpSchema& Input(const int n, const char* name, const char* description)
        {
            return Input(name, description);
        }
        OpSchema& Output(const int n, const char* name, const char* description)
        {
            return Output(name, description);
        }
        OpSchema& SetDoc(const std::string& doc)
        {
            return Description(doc);
        }
#endif // #ifdef ONNX_V1_OPSCHEMA_COMPAT

    private:

        //friend class OperatorSchema;
        friend class OperatorDefinitionRegistry;

        OperatorDefinition m_opDefData;

        // Operator input formal parameters.
        std::vector<InputOutputParam> m_inputs;

        // Operator output formal parameters.
        std::vector<InputOutputParam> m_outputs;

        // Operator type constraints.
        std::vector<TypeConstraintParam> m_constraints;
    };

    // Operator schema registry. A singleton registry to manage all operator
    // schemas.
    class OperatorDefinitionRegistry
    {
    public:

        // Helper function providing a way to call
        // OperatorSchemaFactory::Register().
        class RegisterOnce
        {
        public:

            RegisterOnce(OperatorDefinitionSetter& p_opRegistry);
        };

        // Try to get operator with specified operator name.
        bool TryGetOp(const std::string& p_name,
            const OperatorDefinition** p_opRegistry) const;

        // Register an operator.
        Status Register(const OperatorDefinition& p_opDefData);

        // Get the global operator registry factory instance.
        static OperatorDefinitionRegistry* Get();

    private:

        OperatorDefinitionRegistry() = default;

        // An operator name to operator definition data map.
        std::unordered_map<std::string, OperatorDefinition> m_opNameToOpDefDataMap;
    };

#ifdef ONNX_V1_OPSCHEMA_COMPAT
    // utility function used by ONNX v1 op registration defs.
    size_t ReplaceAll(std::string& s, const char* from, const char* to);
#define OPERATOR_SCHEMA(OpName) OPERATOR_DEFINITION(OpName)
#endif // #ifdef ONNX_V1_OPSCHEMA_COMPAT

#define OPERATOR_DEFINITION(OpName) OPERATOR_DEFINITION_UNIQ_HELPER(__COUNTER__, OpName)
#define OPERATOR_DEFINITION_UNIQ_HELPER(Counter, OpName) OPERATOR_DEFINITION_UNIQ(Counter, OpName)
#define OPERATOR_DEFINITION_UNIQ(Counter, OpName)                     \
    static OperatorDefinitionRegistry::RegisterOnce op_##Counter  \
    = OperatorDefinitionSetter().Name(#OpName)

    // Operator registration example.
    // OPERATOR_DEFINITION(Add).Description("An operator to sum two float numbers.")
    //   .Input("input_1", "docstr for input_1.", "T")
    //   .Input("input_2", "docstr for input_2.", "T")
    //   .Output("output_1", "docstr for output_1.", "T")
    //   .TypeConstraint("T", { "float16", "float32", "float64" }, "Constrain input and output types to floats.");
}

#endif