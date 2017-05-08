/* 
 * File:   simulator.h
 * Author: ryan
 *
 * Created on 26 February 2017, 22:17
 */

#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "graphs/graph_settings.h"

#include "tools/loader.h"
#include "tools/logging.h"

#include "messages/message.h"

#include "states/predict_state.h"
#include "states/forward_train_state.h"

#include "nodes/node.h"
#include "nodes/output_node.h"
#include "nodes/sync_node.h"

#include "training/stochastic_training.h"
#include "training/stochastic_momentum_training.h"

#include "tbb/parallel_for.h"

#include "tools/clock.h"

#include <algorithm>
#include <list>
#include <cassert>
#include <fstream>
#include <memory>
#include <iostream>
#include <cstdio>
#include <stdarg.h> 

using namespace std;

template<typename TNode>
class Simulator{
private:
    friend class Loader;
    
    typedef typename TNode::SyncNode SyncNode;
    
    shared_ptr<GraphSettings> m_settings; 
    vector<Edge*> m_edges;
    vector<Node*> m_nodes;
    multimap<string, Node*> m_node_map;
    
    list<Edge*> activeEdges;
    list<Node*> readyNodes;
    
    std::ostream& m_statsDst;
    int m_step;
    int m_logLevel;
    
    string m_command;
    
    bool step_edge(list<Edge*>::iterator& it){
        Edge* e = (*it);
        
        if(e->msgStatus > 1){
            e->msgStatus = static_cast<Edge::MessageStatus>(int(e->msgStatus)-1);
            it++;
            return true;
        }
        
        e->msg->dispatchTo(e->dst);
        e->msgStatus=Edge::MessageStatus::empty; // The edge is now idle
        it = activeEdges.erase(it);
        
        return true;
    }
    
    bool step_node(unsigned index, Node* n){
        // Not ready to send
        if(!n->readyToSend()){
            Logging::log(4, "node %u : idle", index);
            return false; // Device doesn't want to send
        }
        
        for(auto p: n->outgoingEdges){
            auto e = p.second;
            if( e->msgStatus>0 ){
                Logging::log(3, "node %u : blocked on %u->%u", index, 
                        e->src->getId(),
                        e->dst->getId());
                return true; // One of the outputs is full, so we are blocked
            }
        }
        
        Logging::log(3, "%s node %u : send", n->getType().c_str(), index);
           
        // Get the device to send the message
        vector<Message*> msgs; 
        n->onSend(msgs);
        n->send(msgs, activeEdges);
        
        return true;
    }
    
    bool step_all(){
        Logging::log(2, "stepping edges");
        bool active=false;
        list<Edge*>::iterator it = activeEdges.begin();
        while( it != activeEdges.end()){
            active = step_edge(it) || active;
        }        
        Logging::log(2, "stepping nodes");
        for(unsigned i = 0; i < m_nodes.size(); i++){
            if(step_node(i, m_nodes[i])) active = true;
        }
//        tbb::parallel_for(tbb::blocked_range<size_t>(0,m_nodes.size(),2000),[&](tbb::blocked_range<size_t>& r){
//            for(int i = r.begin(); i != r.end(); ++i)
//                if(step_node(i, m_nodes[i])) active = true;
//        },tbb::auto_partitioner());
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
        
    void setGraphSettings(shared_ptr<GraphSettings> settings){
        m_settings = settings;
    }
        
    void addEdge(int src, int dst, int delay = 1){
        auto e = new Edge(m_nodes[src],m_nodes[dst],delay);
        m_nodes[e->src->getId()]->addEdge(e);
        m_nodes[e->dst->getId()]->addEdge(e);
        m_edges.push_back(e);
    }

    void addNode(Node* node){
        m_node_map.insert(pair<string,Node*>(node->getType(),node));
        m_nodes.push_back(node);
    }
    
    void loadInput(DataWrapper* dataset){
        Logging::log(2, "loading input");
        auto ii = m_node_map.equal_range(SyncNode::m_type);
        int i = 0;
        for(auto it = ii.first; it != ii.second; ++it){
             auto node = dynamic_cast<SyncNode*>(it->second);
             node->setDataSet(dataset);
        }
    }
    
    void run(const string& command){
        Logging::log(1, "begin run");
        bool active=true;
        
        if("predict"==command){
            m_settings->state = new PredictState();
        } else if("train"== command){
            m_settings->state = new ForwardTrainState();
        } else {
            assert(0);
        }
        
        while(active){
            active = step_all();
        }
    }
};

#endif /* SIMULATOR_H */