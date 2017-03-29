/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "misc/backward_propagation_edge.h"

std::string BackwardPropagationEdge::m_type = "BackwardPropagation";
EdgeRegister<BackwardPropagationEdge> BackwardPropagationEdge::m_reg(BackwardPropagationEdge::m_type);