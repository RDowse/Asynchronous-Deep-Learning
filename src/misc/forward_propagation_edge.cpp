/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "misc/forward_propagation_edge.h"

std::string ForwardPropagationEdge::m_type = "ForwardPropagation";
EdgeRegister<ForwardPropagationEdge> ForwardPropagationEdge::m_reg(ForwardPropagationEdge::m_type);