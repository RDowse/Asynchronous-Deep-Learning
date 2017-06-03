/* 
 * File:   bias_node.h
 * Author: ryan
 *
 * Created on 04 April 2017, 22:54
 */

#ifndef PARALLEL_DATA_BIAS_NODE_H
#define PARALLEL_DATA_BIAS_NODE_H

#include "nodes/pardata_nodes/parallel_data_neural_node.h"
#include "misc/node_factory.h"
#include "graphs/graph_settings.h"
#include "graphs/dnn_graph_settings.h"
#include "tools/math.h"

#include <Eigen/StdVector>
#include <cassert>

using namespace std;

class ParallelDataNeuralNode::BiasNode : public ParallelDataNeuralNode{
    // for populating weights map
    int map_index = 0;
    unordered_map<int,int> dstWeightIndex;        // map of weights associated to dst ids
    
    int updateCount = 0;
    
    Eigen::MatrixXf receivedDelta;
    Eigen::VectorXf deltaWeights;
    Eigen::VectorXf weights; 
    
    std::vector<Eigen::VectorXf,Eigen::aligned_allocator<Eigen::VectorXf> > input; // set to 1
public:
    static std::string m_type;
    BiasNode(shared_ptr<GraphSettings> context): ParallelDataNeuralNode(context){
        try{
            auto tmp_context = std::static_pointer_cast<DNNGraphSettings>(context);
            input = vector<Eigen::VectorXf,Eigen::aligned_allocator<Eigen::VectorXf> >(tmp_context->numModels);
        } catch (const std::bad_cast& e) {
            std::cout << e.what() << "\n";
        }
    };
    virtual ~BiasNode(){}
    string getType() override {return BiasNode::m_type;}
    void addEdge(Edge* e) override;
    void setWeights(const vector<float>& w) override{ assert(0); }
    
    void onRecv(ForwardPropagationMessage* msg) override;
    void onRecv(BackwardPropagationMessage* msg) override;
    
    bool sendBackwardMsgs(vector<Message*>& msgs, int stateIndex) override;
    bool sendForwardMsgs(vector<Message*>& msgs, int stateIndex) override;
private:
    void initWeights(){
        weights = Eigen::VectorXf::Zero(outgoingForwardEdges.size());
        context->initWeightsFnc(weights,outgoingForwardEdges.size(),incomingForwardEdges.size());
        
        // init size of delta values
        deltaWeights = Eigen::VectorXf::Zero(weights.size());
    }
};


#endif /* BIAS_NODE_H */

