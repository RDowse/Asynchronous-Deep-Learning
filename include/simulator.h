/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   simulator.h
 * Author: ryan
 *
 * Created on 26 February 2017, 22:17
 */

#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "tools/logging.h"

#include <cassert>
#include <fstream>
#include <memory>
#include <iostream>
#include <cstdio>
#include <stdarg.h> 

class Simulator{
private:
    std::vector<std::shared_ptr<Edge>> m_edges;
    std::vector<std::shared_ptr<Node>> m_nodes;
    
    struct stats
    {
        uint32_t stepIndex;
        
        uint32_t nodeIdleSteps;
        uint32_t nodeBlockedSteps;
        uint32_t nodeSendSteps;
        
        uint32_t edgeIdleSteps;
        uint32_t edgeTransitSteps;
        uint32_t edgeDeliverSteps;
    };
    
    std::ostream& m_statsDst;
    stats m_stats;
    int m_step;
    int m_logLevel;
    
    bool step_edge(unsigned index, std::shared_ptr<Edge>& e){
        if(e->msgStatus == 0){
            Logging::log(4, "  edge %u -> %u : empty", e->src->getId(), e->dst->getId());
            //m_stats.edgeIdleSteps++;
            return false;
        }
        
        if(e->msgStatus > 1){
            Logging::log(3, "  edge %u -> %u : delay (%u)", e->src->getId(), e->dst->getId(), e->msgStatus);
            e->msgStatus = static_cast<Edge::MessageStatus>(int(e->msgStatus)-1);
            //m_stats.edgeTransitSteps++;
            return true;
        }
       
        Logging::log(3, "  edge %u -> %u : deliver", e->src->getId(), e->dst->getId());
        //m_stats.edgeDeliverSteps++;
             
        e->dst->onRecv(e->msg);
        e->msgStatus=Edge::MessageStatus::empty; // The edge is now idle
        
        return true;
    }
    
    bool step_node(unsigned index, std::shared_ptr<Node>& n){
        // Not ready to send
        if(!n->readyToSend()){
            Logging::log(4, "  node %u : idle", index);
            //m_stats.nodeIdleSteps++;
            return false; // Device doesn't want to send
        }
        
        for(unsigned i=0; i < n->outgoingEdges.size(); i++){
            if( n->outgoingEdges[i]->msgStatus>0 ){
                Logging::log(3, "  node %u : blocked on %u->%u", index, 
                        n->outgoingEdges[i]->src->getId(),
                        n->outgoingEdges[i]->src->getId());
                //m_stats.nodeBlockedSteps++;
                return true; // One of the outputs is full, so we are blocked
            }
        }
        
        Logging::log(3, "  node %u : send", index);
        //m_stats.nodeSendSteps++;
        
        auto message = std::make_shared<Message>();
        
        // Get the device to send the message
        n->onSend(message);
        
        // Copy message to edge
        for(unsigned i=0; i < n->outgoingEdges.size(); i++){
            assert( 0 == n->outgoingEdges[i]->msgStatus );
            n->outgoingEdges[i]->msg = message; // Copy message into channel
            n->outgoingEdges[i]->msgStatus = 
                static_cast<Edge::MessageStatus>(1 + n->outgoingEdges[i]->getDelay()); // How long until it is ready?
        }
        
        return true;
    }
    
    bool step_all(){
        Logging::log(2, "stepping edges");
        bool active=false;
        for(unsigned i=0; i<m_edges.size(); i++){
            active = step_edge(i ,m_edges[i]) || active;
        }        
        Logging::log(2, "stepping nodes");
        for(unsigned i=0; i<m_nodes.size(); i++){
            active = step_node(i, m_nodes[i]) || active;
        }
        return active;
    }
    
    void reset(){
        Logging::log(2, "resetting nodes");
        m_step=0;
        Logging::log(2, "resetting edges");
    }
    
public:
    Simulator(int logLevel, 
              std::ostream& stats):
        m_logLevel(logLevel),
        m_step(0),
        m_statsDst(stats){
            Logging::m_logLevel = m_logLevel;
    }
        
    void addEdge(){

    }

    void addNode(){

    }
        
    void run(){
        Logging::log(1, "begin run");
        
        bool active=true;
        
        reset();
        
        while(active){
            active = step_all();
            
            /*
            m_statsDst<<m_stats.stepIndex<<", "<<m_stats.nodeIdleSteps<<", "<<m_stats.nodeBlockedSteps<<", "<<m_stats.nodeSendSteps;
            m_statsDst<<", "<<m_stats.edgeIdleSteps<<", "<<m_stats.edgeTransitSteps<<", "<<m_stats.edgeDeliverSteps<<"\n";
            m_step++;
             */
        }
    }
};

#endif /* SIMULATOR_H */

