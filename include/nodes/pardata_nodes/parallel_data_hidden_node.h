/* 
 * File:   dnn_node.h
 * Author: ryan
 *
 * Created on 20 March 2017, 23:50
 */

#ifndef PARALLEL_DATA_HIDDEN_NODE_H
#define PARALLEL_DATA_HIDDEN_NODE_H

#include "nodes/pardata_nodes/parallel_data_neural_node.h"
#include "misc/node_factory.h"

#include "graphs/dnn_graph_settings.h"
#include "graphs/graph_settings.h"

#include "tools/math.h"

#include <Eigen/StdVector>
#include <eigen3/Eigen/Dense>
#include <string>
#include <cassert>
#include <cstdio>
#include <exception>
#include <memory>
#include <typeinfo>
#include <functional>

using namespace std;

class ParallelDataNeuralNode::HiddenNode: public ParallelDataNeuralNode{// for populating weights map
    int map_index = 0;
    unordered_map<int,int> dstWeightIndex; // map of weights associated to dst ids
    
    int updateCount = 0;
    
    Eigen::MatrixXf receivedDelta;    // store received delta values
    Eigen::VectorXf deltaWeights;     // delta weights, for momentum
    Eigen::VectorXf weights;          // current weights
    
    std::vector<Eigen::VectorXf,Eigen::aligned_allocator<Eigen::VectorXf> > prevWeights;
    std::vector<Eigen::VectorXf,Eigen::aligned_allocator<Eigen::VectorXf> > input;
public:    
    static std::string m_type;
    HiddenNode(shared_ptr<GraphSettings> context): ParallelDataNeuralNode(context){
        try{
            auto tmp_context = std::static_pointer_cast<DNNGraphSettings>(context);
            input = vector<Eigen::VectorXf,Eigen::aligned_allocator<Eigen::VectorXf> >(tmp_context->numModels);
            prevWeights = vector<Eigen::VectorXf,Eigen::aligned_allocator<Eigen::VectorXf> >(tmp_context->numModels);
        } catch (const std::bad_cast& e) {
            std::cout << e.what() << "\n";
        }
    }
    virtual ~HiddenNode(){}
    string getType() override {return HiddenNode::m_type;}
    
    void addEdge(Edge* e) override;
    
    void setWeights(const vector<float>& w) override{ assert(0); }

    void onRecv(ForwardPropagationMessage* msg) override;
    void onRecv(BackwardPropagationMessage* msg) override;

    bool sendForwardMsgs(vector<Message*>& msgs, int stateIndex) override;
    bool sendBackwardMsgs(vector<Message*>& msgs, int stateIndex) override;
private:
    void initWeights(){
        weights = Eigen::VectorXf::Zero(outgoingForwardEdges.size());
        deltaWeights = Eigen::VectorXf::Zero(weights.size());
        context->initWeightsFnc(weights,outgoingForwardEdges.size(),incomingForwardEdges.size());
    }
};

#endif /* HIDDEN_NODE_H */

