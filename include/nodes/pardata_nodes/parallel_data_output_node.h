/* 
 * File:   output_node.h
 * Author: ryan
 *
 * Created on 23 March 2017, 23:54
 */

#ifndef PARALLEL_DATA_OUTPUT_NODE_H
#define PARALLEL_DATA_OUTPUT_NODE_H

#include "nodes/pardata_nodes/parallel_data_neural_node.h"
#include "graphs/graph_settings.h"
#include "misc/node_factory.h"
#include "graphs/dnn_graph_settings.h"
#include "tools/logging.h"

#include <Eigen/StdVector>
#include <cassert>
#include <cstdio>
#include <exception>
#include <memory>
#include <typeinfo>

using namespace std;

class ParallelDataNeuralNode::OutputNode: public ParallelDataNeuralNode{
    std::vector<Eigen::VectorXf,Eigen::aligned_allocator<Eigen::VectorXf> > target;
    std::vector<Eigen::VectorXf,Eigen::aligned_allocator<Eigen::VectorXf> > input;
public:
    static std::string m_type;
    OutputNode(shared_ptr<GraphSettings> context): ParallelDataNeuralNode(context){
        try{
            auto tmp_context = std::static_pointer_cast<DNNGraphSettings>(context);
            input = vector<Eigen::VectorXf,Eigen::aligned_allocator<Eigen::VectorXf> >(tmp_context->numModels);
            target = vector<Eigen::VectorXf,Eigen::aligned_allocator<Eigen::VectorXf> >(tmp_context->numModels);
        } catch (const std::bad_cast& e) {
            std::cout << e.what() << "\n";
        }
    }
    virtual ~OutputNode(){}
    string getType() override{return OutputNode::m_type;}
    
    void addEdge(Edge* e) override;

    void onRecv(ForwardPropagationMessage* msg) override;
    void onRecv(BackwardPropagationMessage* msg) override;
    
    bool sendForwardMsgs(vector<Message*>& msgs, int stateIndex) override;
    bool sendBackwardMsgs(vector<Message*>& msgs, int stateIndex) override;
};

#endif /* OUTPUT_NODE_H */

