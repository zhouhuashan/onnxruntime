#include "op.h"
#include "opschema.h"
#include "utils.h"

namespace LotusIR
{
    const std::string& OperatorDefinition::GetName() const
    {
        return m_opSchema.GetName();
    }

    const OperatorSchema& OperatorDefinition::GetOpSchema() const
    {
        return m_opSchema;
    }

    ShapeInferenceFunc OperatorDefinition::GetShapeInferenceFn() const
    {
        return m_shapeInferenceFunc;
    }

    AttributeParser OperatorDefinition::GetAttributeParser() const
    {
        return m_attrParser;
    }

    OperatorDefinitionSetter&
        OperatorDefinitionSetter::Name(const std::string& p_opName)
    {
        m_opDefData.m_opSchema.m_name = p_opName;
        return *this;
    }



    OperatorDefinitionSetter&
        OperatorDefinitionSetter::Description(const std::string& p_description)
    {
        m_opDefData.m_opSchema.m_description = p_description;
        return *this;
    }

    OperatorDefinitionSetter&
        OperatorDefinitionSetter::Input(const std::string& p_inputName,
            const std::string& p_description,
            const std::string& p_type)
    {
        m_inputs.push_back(std::make_tuple(p_inputName, p_description, p_type));
        return *this;
    }

    OperatorDefinitionSetter&
        OperatorDefinitionSetter::Output(const std::string& p_outputName,
            const std::string& p_description,
            const std::string& p_type)
    {
        m_outputs.push_back(std::make_tuple(p_outputName, p_description, p_type));
        return *this;
    }

    OperatorDefinitionSetter&
        OperatorDefinitionSetter::Attr(const std::string& p_attrName,
            const std::string& p_description,
            AttrType p_attrType, bool required)
    {
        m_opDefData.m_opSchema.m_attributes.push_back(
            OperatorSchema::Attribute(p_attrName, p_attrType, p_description));

        return *this;
    }

#define ATTR_SETTER_BASIC_IMPL(type, field)                                               \
    OperatorDefinitionSetter&                                                         \
        OperatorDefinitionSetter::Attr(const std::string& p_attrName,                 \
            const std::string& p_description,                                             \
            AttrType p_attrType,                                                          \
            const type& p_defaultValue)                                                   \
    {                                                                                     \
        AttributeProto a;                                                                 \
        a.set_name(p_attrName);                                                           \
        a.set_##field(p_defaultValue);                                                    \
                                                                                          \
        m_opDefData.m_opSchema.m_attributes.push_back(                                    \
            OperatorSchema::Attribute(p_attrName,                                         \
                                        p_attrType,                                       \
                                        p_description,                                    \
                                        a));                                              \
                                                                                          \
        return *this;                                                                     \
    }                                                                                     \

#define ATTR_SETTER_LIST_IMPL(type, field)                                                \
    OperatorDefinitionSetter&                                                         \
        OperatorDefinitionSetter::Attr(const std::string& p_attrName,                 \
            const std::string& p_description,                                             \
            AttrType p_attrType,                                                          \
            const std::vector<type>& p_defaultValue)                                      \
    {                                                                                     \
        AttributeProto a;                                                                 \
        a.set_name(p_attrName);                                                           \
        for (const auto& v : p_defaultValue)                                              \
        {                                                                                 \
            a.add_##field(v);                                                             \
        }                                                                                 \
                                                                                          \
        m_opDefData.m_opSchema.m_attributes.push_back(                                    \
        OperatorSchema::Attribute(p_attrName,                                             \
            p_attrType,                                                                   \
            p_description,                                                                \
            a));                                                                          \
        return *this;                                                                     \
    }                                                                                     \

    ATTR_SETTER_BASIC_IMPL(int64_t, i)
    ATTR_SETTER_BASIC_IMPL(float, f)
    ATTR_SETTER_BASIC_IMPL(std::string, s)
    ATTR_SETTER_LIST_IMPL(int64_t, ints)
    ATTR_SETTER_LIST_IMPL(float, floats)
    ATTR_SETTER_LIST_IMPL(std::string, strings)

    OperatorDefinitionSetter&
    OperatorDefinitionSetter::TypeConstraint(const std::string& p_typeName,
        const std::vector<std::string>& p_constraints,
        const std::string& p_description)
    {
        m_constraints.push_back(std::make_tuple(p_typeName, p_constraints, p_description));
        return *this;
    }

    OperatorDefinitionSetter&
        OperatorDefinitionSetter::SetShapeInferenceFunc(
            ShapeInferenceFunc p_shapeInferFunc)
    {
        m_opDefData.m_shapeInferenceFunc = p_shapeInferFunc;
        return *this;
    }

    OperatorDefinitionSetter&
        OperatorDefinitionSetter::SetAttributeParser(
            AttributeParser p_attrParser)
    {
        m_opDefData.m_attrParser = p_attrParser;
        return *this;
    }

    OperatorDefinitionRegistry::RegisterOnce::RegisterOnce(
        OperatorDefinitionSetter& p_opDefDataSetter)
    {
        auto& opDefData = p_opDefDataSetter.m_opDefData;
        // Process type constraints.
        for (const auto& constraint : p_opDefDataSetter.m_constraints)
        {
            std::string name;
            std::vector<std::string> types;
            std::string desc;
            std::tie(name, types, desc) = constraint;

            auto it = opDefData.m_opSchema.m_typeConstraintMap.find(name);
            if (it == opDefData.m_opSchema.m_typeConstraintMap.end())
            {
                DataTypeSet d;
                for (const auto& t : types)
                {
                    d.insert(Utils::OpUtils::ToType(t));
                }
                opDefData.m_opSchema.m_typeConstraintMap.insert(std::make_pair(name, std::make_pair(d, desc)));
            }
            else
            {
                // already a constraint with the same name. error.
            }
        }

        opDefData.m_opSchema.m_inputs.reserve(p_opDefDataSetter.m_inputs.size());
        for (const auto& input : p_opDefDataSetter.m_inputs)
        {
            std::string name;
            std::string type;
            std::string desc;
            std::tie(name, desc, type) = input;
            opDefData.m_opSchema.m_inputs.push_back(
                OperatorSchema::FormalParameter(name, type, desc, opDefData.m_opSchema.m_typeConstraintMap));
        }

        opDefData.m_opSchema.m_outputs.reserve(p_opDefDataSetter.m_outputs.size());
        for (const auto& output : p_opDefDataSetter.m_outputs)
        {
            std::string name;
            std::string type;
            std::string desc;
            std::tie(name, desc, type) = output;
            opDefData.m_opSchema.m_outputs.push_back(
                OperatorSchema::FormalParameter(name, type, desc,
                    opDefData.m_opSchema.m_typeConstraintMap));
        }

#ifdef ONNX_V1_OPSCHEMA_COMPAT
        auto& opSchema = p_opDefDataSetter.m_opDefData.m_opSchema;
        if (0 == opSchema.m_inputs.size())
        {
            for (int i = 0; i < opSchema.m_onnxMinInput; ++i)
            {
                std::string name = "p" + std::to_string(i);
                std::string desc = "Input Parameter " + std::to_string(i);
                opSchema.m_inputs.push_back(
                    OperatorSchema::FormalParameter(name, "", desc, opSchema.m_typeConstraintMap));
            }
        }

        if (0 == opSchema.m_outputs.size())
        {
            for (int i = 0; i < opSchema.m_onnxMinOutput; ++i)
            {
                std::string name = "p" + std::to_string(i);
                std::string desc = "Output Result " + std::to_string(i);
                opSchema.m_outputs.push_back(
                    OperatorSchema::FormalParameter(name, "", desc, opSchema.m_typeConstraintMap));
            }
        }
#endif
        OperatorDefinitionRegistry::Get()->Register(p_opDefDataSetter.m_opDefData);
    }

    bool OperatorDefinitionRegistry::TryGetOp(const std::string& p_name,
        const OperatorDefinition** p_opDefData) const
    {
        if (nullptr == p_opDefData)
        {
            return false;
        }

        auto iter = m_opNameToOpDefDataMap.find(p_name);
        if (m_opNameToOpDefDataMap.end() == iter)
        {
            return false;
        }
        *p_opDefData = &(iter->second);
        return true;
    }

    Status OperatorDefinitionRegistry::Register(
        const OperatorDefinition& p_opDefData)
    {
        auto iter = m_opNameToOpDefDataMap.find(p_opDefData.GetName());
        if (m_opNameToOpDefDataMap.end() != iter)
        {
            Status status(false,
                "Error: operator schema with same name ("
                + p_opDefData.GetName() + ") exists.");
            return status;
        }
        else
        {
            m_opNameToOpDefDataMap[p_opDefData.GetName()] = p_opDefData;
            return Status::OK();
        }
    }

    OperatorDefinitionRegistry* OperatorDefinitionRegistry::Get()
    {
        static OperatorDefinitionRegistry* s_registry
            = new OperatorDefinitionRegistry();
        return s_registry;
    }

    Status TypeUtils::GetType(const AttributeProto& p_attr, AttrType& p_type)
    {
        if (!OperatorSchema::IsValidAttribute(p_attr))
        {
            return Status(false, "Invalid AttributeProto.");
        }

        if (p_attr.has_f())
        {
            p_type = AttrType::FLOAT;
        }
        else if (p_attr.has_i())
        {
            p_type = AttrType::INT;
        }
        else if (p_attr.has_s())
        {
            p_type = AttrType::STRING;
        }
        else if (p_attr.has_t())
        {
            p_type = AttrType::TENSOR;
        }
        else if (p_attr.has_g())
        {
            p_type = AttrType::GRAPH;
        }
        else if (p_attr.floats_size())
        {
            p_type = AttrType::FLOATS;
        }
        else if (p_attr.ints_size())
        {
            p_type = AttrType::INTS;
        }
        else if (p_attr.strings_size())
        {
            p_type = AttrType::STRINGS;
        }
        else if (p_attr.tensors_size())
        {
            p_type = AttrType::TENSORS;
        }
        else if (p_attr.graphs_size())
        {
            p_type = AttrType::GRAPHS;
        }
        else if (p_attr.has_type())
        {
            p_type = AttrType::TYPE;
        }
        else if (p_attr.types_size())
        {
            p_type = AttrType::TYPES;
        }
        else if (p_attr.has_shape())
        {
            p_type = AttrType::SHAPE;
        }
        else if (p_attr.has_shape())
        {
            p_type = AttrType::SHAPES;
        }
        else
        {
            p_type = AttrType::NONE;
            return Status(false, "Invalid AttributeProto.");
        }

        return Status::OK();
    }

#ifdef ONNX_V1_OPSCHEMA_COMPAT
    size_t ReplaceAll(std::string& s, const char* from, const char* to)
    {
        size_t numReplaced = 0;
        std::string::size_type lenFrom = std::strlen(from);
        std::string::size_type lenTo = std::strlen(to);
        for (std::string::size_type pos = s.find(from); pos != std::string::npos;
            pos = s.find(from, pos + lenTo)) {
            s.replace(pos, lenFrom, to);
            numReplaced++;
        }
        return numReplaced;
    }
#endif
}