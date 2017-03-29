/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   forward_propagation_edge.h
 * Author: ryan
 *
 * Created on 29 March 2017, 19:18
 */

#ifndef FORWARD_PROPAGATION_EDGE_H
#define FORWARD_PROPAGATION_EDGE_H

#include "misc/edge.h"
#include "messages/forward_propagation_message.h"
#include "misc/edge_factory.h"

class ForwardPropagationEdge: public Edge{
    static EdgeRegister<ForwardPropagationEdge> m_reg;
    static std::string m_type;
public:
    shared_ptr<ForwardPropagationMessage> msg; 
    ForwardPropagationEdge(shared_ptr<Node> dst, shared_ptr<Node> src, unsigned delay) :
    Edge(dst,src,delay) {}
};

#endif /* FORWARD_PROPAGATION_EDGE_H */

