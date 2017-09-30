#include "shape_inference.h"

namespace LotusIR
{
    InferenceContext::InferenceContext(Node* p_node,
        const OperatorSchema* p_opSchema)
        : m_node(p_node),
        m_opSchema(p_opSchema)
    {
    }

    const Node* InferenceContext::GetNode() const
    {
        return m_node;
    }

    const OperatorSchema* InferenceContext::GetOp() const
    {
        return m_opSchema;
    }

    const std::vector<NodeArg>* InferenceContext::GetInputs() const
    {
        if (nullptr == m_node)
        {
            return nullptr;
        }
        return &(m_node->InputDefs());
    }

    std::vector<NodeArg>* InferenceContext::Mutable_Outputs()
    {
        if (nullptr == m_node)
        {
            return nullptr;
        }
        return &(m_node->Mutable_OutputDefs());
    }
}