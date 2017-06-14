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
#include "tools/strategy_loader.h"

#include "messages/message.h"

#include "states/predict_state.h"
#include "states/forward_train_state.h"

#include "nodes/node.h"
#include "nodes/output_node.h"
#include "nodes/sync_node.h"

#include "nodes/pardata_nodes/parallel_data_neural_node.h"
#include "nodes/pardata_nodes/parallel_data_sync_node.h"
#include "nodes/pardata_nodes/parallel_data_output_node.h"

#include "tbb/parallel_for.h"
#include "tbb/concurrent_queue.h"
#include "tbb/tick_count.h"
#include "tbb/concurrent_vector.h"

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
    
    shared_ptr<GraphSettings> context; 
    vector<Edge*> m_edges;
    vector<Node*> m_nodes;
    multimap<string, Node*> m_node_map;
    
    list<Edge*> activeEdges;
    
    std::ostream& m_statsDst;
    int m_step;
    int m_logLevel;
    
    string m_command;
    
    bool step_edge(Edge *e){
    
        if(e->msgStatus == 0){
            return false; // something went wrong
        }
        
        if(e->msgStatus > 1){
            e->msgStatus--;
            return true;
        }
        
        e->msg->dispatchTo(e->dst);
        e->msgStatus=0; // The edge is now idle
        
        return true;
    }
    
    bool step_edge(list<Edge*>::iterator& it){
        Edge* e = (*it);
        
        if(e->msgStatus == 0){
            assert(0); // something went wrong
        }
        
        if(e->msgStatus > 1){
            e->msgStatus--;
            it++;
            return true;
        } 
        
        e->msg->dispatchTo(e->dst);
        e->msgStatus=0; // The edge is now idle
        it = activeEdges.erase(it);
        
        return true;
    }
    
    bool step_node(Node* n){
        // Not ready to send
        if(!n->readyToSend()){
            Logging::log(4, "node %u : idle", n->getId());
            return false; // Device doesn't want to send
        }
        
        for(auto p: n->outgoingEdges){
            auto e = p.second;
            if( e->msgStatus > 0 ){
                Logging::log(1, "node %u : blocked on %u->%u", n->getId(), 
                        e->src->getId(),
                        e->dst->getId());
                return true; // One of the outputs is full, so we are blocked
            }
        }
        
        Logging::log(3, "%s node %u : send", n->getType().c_str(), n->getId());
           
        // Get the device to send the message
        vector<Message*> msgs; 
        n->onSend(msgs);
        n->send(msgs, activeEdges);
        
        return true;
    }
    
    void step_node_par(Node* n, bool& active){
        // Not ready to send
        if(!n->readyToSend()){
            Logging::log(4, "node %u : idle", n->getId());
            return; // Device doesn't want to send
        }
        
        for(auto p: n->outgoingEdges){
            auto e = p.second;
            if( e->msgStatus > 0 ){
                Logging::log(1, "node %u : blocked on %u->%u", n->getId(), 
                        e->src->getId(),
                        e->dst->getId());
                active = true;
                return; // One of the outputs is full, so we are blocked
            }
        }
        
        Logging::log(3, "%s node %u : send", n->getType().c_str(), n->getId());
           
        // Get the device to send the message
        vector<Message*> msgs; 
        n->onSend(msgs);
        n->send(msgs);
        
        active = true;
        return;
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
            active = step_node(m_nodes[i]) || active;
        }
        return active;
    }
    
    bool step_all_parallel(){
        
        Logging::log(2, "stepping edges");
        bool active=false;    
        for(auto& e: m_edges){
            active = step_edge(e) || active;
        }
        
        Logging::log(2, "stepping nodes");
        
        tbb::parallel_for(tbb::blocked_range<std::vector<Node*>::iterator>(m_nodes.begin(),m_nodes.end()),
            [&] (tbb::blocked_range<std::vector<Node*>::iterator> node) {
            for (std::vector<Node*>::iterator it = node.begin(); it != node.end(); it++) {
                step_node_par(*it,active);
            }
        },tbb::auto_partitioner());
        
        // global time for async
        context->incrementTime();
        
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
//        for(auto it = m_nodes.begin(); it != m_nodes.end(); it++)
//            delete (*it);
        for(int i = 0; i < m_nodes.size(); i++)
            delete m_nodes[i];
        m_nodes.clear();
//        for(auto it = m_edges.begin(); it != m_edges.end(); it++)
//            delete (*it);
        for(int i = 0; i < m_edges.size(); i++)
            delete m_edges[i];
        m_edges.clear();
    }
    
    // strategy config from YAML file
    void setStrategies(vector<string> config){
        StrategyLoader<TNode> stratLoader(context,&m_nodes);
        stratLoader.setConfig(config);
        stratLoader.load();
    }
        
    template<typename TState>
    void setState(){
        for(auto n: m_nodes)
            dynamic_cast<TNode*>(n)->setState(new TState());
    }
    
    void setGraphSettings(shared_ptr<GraphSettings> settings){
        context = settings;
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
    
    double edgeModTimeTotal = 0;
    void modifyEdgeDelay(){
        tbb::tick_count t0 = tbb::tick_count::now();
        
        for(auto e: m_edges)
            e->delay = context->delayInitialiser();
        
        edgeModTimeTotal += (tbb::tick_count::now()-t0).seconds();
    }
    
    // For results gathering, specific to the node type
    void postProcessing(shared_ptr<GraphSettings> _context){}
    
    void run(const string& command){
        Logging::log(1, "begin run");
        bool active=true;
        
        if("predict"==command){
            setState<PredictState<TNode>>();
        } else if("train"== command){
            setState<ForwardTrainState<TNode>>();
        } else {
            assert(0);
        }
        
        modifyEdgeDelay();
        
        tbb::tick_count t0 = tbb::tick_count::now();
        while(active){
            active = step_all_parallel();
            
            // adjust the delay of edges
            if(context->enableVariableEdgeDelay && (context->stepTime % 25 == 0) ) modifyEdgeDelay();
        }
        tbb::tick_count t1 = tbb::tick_count::now();
        cout << "Total run time: " << (t1-t0).seconds() - edgeModTimeTotal << endl;
        context->runTime = (t1-t0).seconds() - edgeModTimeTotal;
        
        cout << "Total steps: " << context->stepTime << endl;
        
        postProcessing(context);
    }
};

template<> void Simulator<AsyncNeuralNode>::postProcessing(shared_ptr<GraphSettings> _context){
    cout << "Starting post processing of result" << endl;
    try{
        auto context = std::static_pointer_cast<DNNGraphSettings>(_context);
        for(auto n: m_nodes){
        if(auto node = dynamic_cast<AsyncNeuralNode::HiddenNode*>(n)){
            context->numBackwardMessagesDropped += node->discardedBackwardMessageCount;
            context->numForwardMessagesDropped += node->discardedForwardMessageCount;
            context->numBackwardMessagesSent += node->numMessagesSentBackward;
            context->numForwardMessagesSent += node->numMessagesSentForward;
        } else if(auto node = dynamic_cast<AsyncNeuralNode::BiasNode*>(n)){
            context->numBackwardMessagesDropped += node->discardedBackwardMessageCount;
            context->numForwardMessagesDropped += node->discardedForwardMessageCount;
            context->numBackwardMessagesSent += node->numMessagesSentBackward;
            context->numForwardMessagesSent += node->numMessagesSentForward;    
        } else if(auto node = dynamic_cast<AsyncNeuralNode::InputNode*>(n)){
            context->numBackwardMessagesDropped += node->discardedBackwardMessageCount;
            context->numForwardMessagesDropped += node->discardedForwardMessageCount;
            context->numBackwardMessagesSent += node->numMessagesSentBackward;
            context->numForwardMessagesSent += node->numMessagesSentForward;
        } else if(auto node = dynamic_cast<AsyncNeuralNode::OutputNode*>(n)){
            context->numBackwardMessagesDropped += node->discardedBackwardMessageCount;
            context->numForwardMessagesDropped += node->discardedForwardMessageCount;
            context->numBackwardMessagesSent += node->numMessagesSentBackward;
            context->numForwardMessagesSent += node->numMessagesSentForward;
        } else if(auto node = dynamic_cast<AsyncNeuralNode::SyncNode*>(n)){
            context->numBackwardMessagesDroppedSync += node->discardedBackwardMessageCount;
            context->numForwardMessagesDroppedSync += node->discardedForwardMessageCount;
            context->numBackwardMessagesSentSync += node->numMessagesSentBackward;
            context->numForwardMessagesSentSync += node->numMessagesSentForward;
        }
    }
    } catch (const std::bad_cast& e) {
        std::cout << e.what() << "\n";
        exit(1);
    }

}

#endif /* SIMULATOR_H */