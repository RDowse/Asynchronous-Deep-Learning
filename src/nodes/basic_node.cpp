/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "nodes/basic_node.h"

std::string BasicNode::m_type = "Basic";
NodeRegister<BasicNode> BasicNode::m_reg(BasicNode::m_type);