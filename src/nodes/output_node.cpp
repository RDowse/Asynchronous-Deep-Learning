
#include "nodes/output_node.h"

std::string OutputNode::m_type = "Output";
NodeRegister<OutputNode> OutputNode::m_reg(OutputNode::m_type);