/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "nodes/dnn_node.h"

std::string DNNNode::m_type = "DNN";
NodeRegister<DNNNode> DNNNode::m_reg(DNNNode::m_type);