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

#include "tools/loader.h"
#include "tools/logging.h"

#include "messages/message.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"

#include "nodes/node.h"
#include "nodes/output_node.h"
#include "nodes/sync_node.h"

#include "graphs/graph_settings.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <memory>
#include <iostream>
#include <cstdio>
#include <stdarg.h> 

using namespace std;

class Simulator{
private:
    friend class Loader;
    
    shared_ptr<DNNGraphSettings> m_settings; 
    vector<Edge*> m_edges;
    vector<Node*> m_nodes;
    multimap<string, Node*> m_node_map;
    
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
    
    //std::map<string,function> m_cmd_map;
    string m_command;
    
    bool step_edge(unsigned index, Edge* e){
        if(e->msgStatus == 0){
            Logging::log(4, "  edge %u -> %u : empty", e->src->getId(), e->dst->getId());
            //m_stats.edgeIdleSteps++;
            return false;
        }
        
        if(e->msgStatus > 1){
            //Logging::log(3, "  edge %u -> %u : delay (%u)", e->src->getId(), e->dst->getId(), e->msgStatus);
            e->msgStatus = static_cast<Edge::MessageStatus>(int(e->msgStatus)-1);
            //m_stats.edgeTransitSteps++;
            return true;
        }
       
        //Logging::log(3, "  edge %u -> %u : deliver", e->src->getId(), e->dst->getId());
        //m_stats.edgeDeliverSteps++;
        e->msg->dispatchTo(e->dst);
        e->msgStatus=Edge::MessageStatus::empty; // The edge is now idle
        
        return true;
    }
    
    bool step_node(unsigned index, Node* n){
        // Not ready to send
        if(!n->readyToSend()){
            Logging::log(4, "node %u : idle", index);
            //m_stats.nodeIdleSteps++;
            return false; // Device doesn't want to send
        }
        
        for(unsigned i=0; i < n->outgoingEdges.size(); i++){
            if( n->outgoingEdges[i]->msgStatus>0 ){
                Logging::log(3, "node %u : blocked on %u->%u", index, 
                        n->outgoingEdges[i]->src->getId(),
                        n->outgoingEdges[i]->src->getId());
                //m_stats.nodeBlockedSteps++;
                return true; // One of the outputs is full, so we are blocked
            }
        }
        
        Logging::log(3, "%s node %u : send", n->getType().c_str(), index);
        //m_stats.nodeSendSteps++;
           
        // Get the device to send the message
        vector<shared_ptr<Message> > msgs; 
        n->onSend(msgs);
        
//        for(unsigned i=0; i < msgs.size(); i++){
//            assert( 0 == n->outgoing[i]->messageStatus );
//            forwardEdges[i]->msg = msg; // Copy message into channel
//            forwardEdges[i]->msgStatus = 
//                static_cast<Edge::MessageStatus>(1 + forwardEdges[i]->getDelay()); // How long until it is ready?
//        }
        
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
              unsigned nNodes,
              unsigned nEdges,
              std::ostream& stats):
        m_logLevel(logLevel),
        m_step(0),
        m_statsDst(stats){
        Logging::m_logLevel = m_logLevel;
        m_nodes.reserve(nNodes);
        m_edges.reserve(nEdges);
    }
        
    ~Simulator(){
        for(auto it = m_nodes.begin(); it != m_nodes.end(); it++)
            delete (*it);
        m_nodes.clear();
        for(auto it = m_edges.begin(); it != m_edges.end(); it++)
            delete (*it);
        m_edges.clear();
    }
        
    void setGraphSettings(shared_ptr<DNNGraphSettings> settings){
        m_settings = settings;
    }
        
    void addEdge(int src, int dst, int delay){
        auto e = new Edge(m_nodes[src],m_nodes[dst],delay);
        m_nodes[src]->outgoingEdges.push_back(e);
        m_nodes[dst]->incomingEdges.push_back(e);
        m_edges.push_back(e);
    }

    void addNode(Node* node){
        m_node_map.insert(pair<string,Node*>(node->getType(),node));
        m_nodes.push_back(node);
    }
    
    void loadInput(DataWrapper* dataset){
        Logging::log(2, "loading input");
        auto ii = m_node_map.equal_range("Sync");
        int i = 0;
        for(auto it = ii.first; it != ii.second; ++it){
             auto node = dynamic_cast<SyncNode*>(it->second);
             node->setDataSet(dataset);
        }
    }
    
    void setup(){
        for(auto n: m_nodes)
            n->setup();
    }
        
    void run(const string& command){
        Logging::log(1, "begin run");
        
        bool active=true;
        if("predict"==command){
            m_settings->cmd = static_cast<DNNGraphSettings::Command>(0);
        } else if("train"==command){
            m_settings->cmd = static_cast<DNNGraphSettings::Command>(1);
        }
        
        while(active){
            active = step_all();
        }
    }
};

#endif /* SIMULATOR_H */