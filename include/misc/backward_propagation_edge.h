/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   backward_propagation_edge.h
 * Author: ryan
 *
 * Created on 29 March 2017, 19:19
 */

#ifndef BACKWARD_PROPAGATION_EDGE_H
#define BACKWARD_PROPAGATION_EDGE_H

#include "misc/edge.h"
#include "messages/backward_propagation_message.h"
#include "misc/edge_factory.h"

class BackwardPropagationEdge: public Edge{
    static EdgeRegister<BackwardPropagationEdge> m_reg;
    static std::string m_type;
public:
    shared_ptr<BackwardPropagationMessage> msg; 
    BackwardPropagationEdge(shared_ptr<Node> dst, shared_ptr<Node> src, unsigned delay):
    Edge(dst,src,delay) {}
    
    std::string getType() const{return BackwardPropagationEdge::m_type;}
};


#endif /* BACKWARD_PROPAGATION_EDGE_H */

