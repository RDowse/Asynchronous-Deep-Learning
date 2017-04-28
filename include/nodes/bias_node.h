/* 
 * File:   bias_node.h
 * Author: ryan
 *
 * Created on 04 April 2017, 22:54
 */

#ifndef BIAS_NODE_H
#define BIAS_NODE_H

#include "nodes/neural_node.h"
#include "misc/node_factory.h"
#include "graphs/graph_settings.h"
#include "graphs/dnn_graph_settings.h"
#include "tools/math.h"

#include <stack>
#include <cassert>

using namespace std;

class BiasNode : public NeuralNode{
    static NodeRegister<BiasNode> m_reg;
    static std::string m_type;
    
    vector<float> deltas;
    vector<float> newWeights;
    vector<float> deltaWeights;
    vector<float> weights; 
public:
    BiasNode(shared_ptr<GraphSettings> context): NeuralNode(context){ output = 0;};
    virtual ~BiasNode(){}
    string getType() override {return BiasNode::m_type;}
    
    void addEdge(Edge* e) override{
        // add to original edge sets
        Node::addEdge(e);
        // check edge belongs to this node
        if(e->src->getId() == m_id){
            if(e->dst->getType() == "Sync"){
                outgoingBackwardEdges.push_back(e);
            } else if(e->dst->getType() == "Hidden"
                    || e->dst->getType() == "Output"){
                outgoingForwardEdges.push_back(e);
            } else {
                cout << "Unknown type " << e->dst->getType() << endl;
                assert(0);
            }
        } else if(e->dst->getId() == m_id){
            if(e->src->getType() == "Sync"){
                incomingForwardEdges.push_back(e);
            } else if(e->src->getType() == "Hidden"
                    || e->src->getType() == "Output"){
                incomingBackwardEdges.push_back(e);
            } else {
                cout << "Unknown type " << e->src->getType() << endl;
                assert(0);
            }
        } 
    }
    
    void setWeights(const vector<float>& w) override{
        assert(w.size() == 1);
        weights = w;
        newWeights = w; 
        
        // init size of delta values
        deltas = vector<float>(weights.size());
        deltaWeights = vector<float>(weights.size());
    }
    
    void onRecv(ForwardPropagationMessage* msg) override;
    void onRecv(BackwardPropagationMessage* msg) override;
    
    bool sendBackwardMsgs(vector<Message*>& msgs) override;
    bool sendForwardMsgs(vector<Message*>& msgs) override;
private:
    // for populating weights map
    int map_index = 0;
    void initWeights(){
        weights = vector<float>(1);
        settings->initWeightsFnc(weights,outgoingForwardEdges.size(),incomingForwardEdges.size());
        newWeights = weights;    
        
        // init size of delta values
        deltas = vector<float>(weights.size());
        deltaWeights = vector<float>(weights.size());
    }
};


#endif /* BIAS_NODE_H */

