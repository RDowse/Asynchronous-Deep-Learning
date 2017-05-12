/* 
 * File:   output_node.h
 * Author: ryan
 *
 * Created on 23 March 2017, 23:54
 */

#ifndef OUTPUT_NODE_H
#define OUTPUT_NODE_H

#include "nodes/neural_node.h"
#include "graphs/graph_settings.h"
#include "misc/node_factory.h"
#include "graphs/dnn_graph_settings.h"
#include "tools/logging.h"

#include <string>
#include <cassert>
#include <cstdio>
#include <exception>
#include <memory>
#include <typeinfo>

using namespace std;

class NeuralNode::OutputNode: public NeuralNode{
    static NodeRegister<OutputNode> m_reg;
    static std::string m_type;
    
    Eigen::VectorXf error;
    Eigen::VectorXf target;
    Eigen::VectorXf value;
public:
    OutputNode(shared_ptr<GraphSettings> context): NeuralNode(context){}
    virtual ~OutputNode(){}
    string getType() override{return OutputNode::m_type;}
    
    void addEdge(Edge* e) override;

    void onRecv(ForwardPropagationMessage* msg) override;
    void onRecv(BackwardPropagationMessage* msg) override;
    
    bool sendForwardMsgs(vector<Message*>& msgs) override;
    bool sendBackwardMsgs(vector<Message*>& msgs) override;
};

#endif /* OUTPUT_NODE_H */

