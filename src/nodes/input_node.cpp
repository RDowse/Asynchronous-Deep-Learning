/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "nodes/input_node.h"

std::string InputNode::m_type = "Input";
NodeRegister<InputNode> InputNode::m_reg(InputNode::m_type);