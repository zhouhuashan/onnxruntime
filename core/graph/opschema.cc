#include "opschema.h"

namespace LotusIR
{
    OperatorSchema::FormalParameter::FormalParameter(
        const std::string& p_name, const std::string& p_type,
        const std::string& p_description,
        const TypeConstraintMap& p_constraintMap)
        : m_name(p_name), m_typeStr(p_type), m_description(p_description)
    {
        auto it = p_constraintMap.find(p_type);
        if (it != p_constraintMap.end())
        {
            m_types = it->second.first;
        }
        else
        {
            if (!p_type.empty())
            {
                m_types.emplace(Utils::OpUtils::ToType(m_typeStr));
            }
        }
    }

    const std::string& OperatorSchema::FormalParameter::GetName() const
    {
        return m_name;
    }

    const DataTypeSet& OperatorSchema::FormalParameter::GetTypes() const
    {
        return m_types;
    }

    const std::string& OperatorSchema::FormalParameter::GetTypeStr() const
    {
        return m_typeStr;
    }

    const std::string& OperatorSchema::FormalParameter::GetDescription() const
    {
        return m_description;
    }

    OperatorSchema::Attribute::Attribute(
        const std::string& p_attrName,
        AttrType p_type,
        const std::string& p_description,
        const AttributeProto& p_defaultVal)
        : m_name(p_attrName), m_type(p_type), m_description(p_description),
        m_hasDefaultValue(true)
    {
        m_allowedValues.push_back(p_defaultVal);
    }

    OperatorSchema::Attribute::Attribute(
        const std::string& p_attrName,
        AttrType p_type,
        const std::string& p_description)
        : m_name(p_attrName), m_type(p_type), m_description(p_description),
        m_hasDefaultValue(false)
    {
    }

    const std::string& OperatorSchema::Attribute::GetName() const
    {
        return m_name;
    }

    AttrType OperatorSchema::Attribute::GetType() const
    {
        return m_type;
    }

    bool OperatorSchema::Attribute::HasDefaultValue(
        const AttributeProto** p_value) const
    {
        if (m_hasDefaultValue
            && nullptr != p_value)
        {
            *p_value = &(m_allowedValues[0]);
        }

        return m_hasDefaultValue;
    }



    const std::string& OperatorSchema::GetName() const
    {
        return m_name;
    }

    const std::string& OperatorSchema::GetDescription() const
    {
        return m_description;
    }

    const std::vector<OperatorSchema::FormalParameter>&
        OperatorSchema::GetInputs() const
    {
        return m_inputs;
    }

    const std::vector<OperatorSchema::FormalParameter>&
        OperatorSchema::GetOutputs() const
    {
        return m_outputs;
    }

    const std::vector<OperatorSchema::Attribute>&
        OperatorSchema::GetAttributes() const
    {
        return m_attributes;
    }

    const TypeConstraintMap& OperatorSchema::GetTypeConstraintMap() const
    {
        return m_typeConstraintMap;
    }

    bool OperatorSchema::IsValidAttribute(const AttributeProto& p_attr)
    {
        if (p_attr.name().empty())
        {
            return false;
        }

        int num_fields =
            p_attr.has_f() +
            p_attr.has_i() +
            p_attr.has_s() +
            p_attr.has_t() +
            p_attr.has_g() +
            (p_attr.floats_size() > 0) +
            (p_attr.ints_size() > 0) +
            (p_attr.strings_size() > 0) +
            (p_attr.tensors_size() > 0) +
            (p_attr.graphs_size() > 0) +
            p_attr.has_type() +
            (p_attr.types_size() > 0) +
            p_attr.has_shape() +
            (p_attr.shapes_size() > 0);

        if (num_fields == 1)
        {
            return true;
        }
        return false;
    }
}