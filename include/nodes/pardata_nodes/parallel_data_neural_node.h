/* 
 * File:   neural_node.h
 * Author: ryan
 *
 * Created on 24 April 2017, 00:27
 */

#ifndef PARALLEL_DATA_NEURAL_NODE_H
#define PARALLEL_DATA_NEURAL_NODE_H

#include "nodes/node.h"
#include "graphs/dnn_graph_settings.h"
#include "misc/message_pool.h"

#include "training/dropout_strategy.h"
#include "training/dropout_null.h"

#include "states/backward_train_state.h"
#include "states/forward_train_state.h"

#include "common.h"

#include <Eigen/StdVector>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <memory>

using namespace std;

template<typename TNode> class State;
class ForwardPropagationMessage;
class BackwardPropagationMessage;

class ParallelDataNeuralNode: public Node{
public:
    class InputNode;
    class HiddenNode;
    class OutputNode;
    class BiasNode;
    class SyncNode;
protected:
    static MessagePool<ForwardPropagationMessage>* forwardMessagePool;
    static MessagePool<BackwardPropagationMessage>* backwardMessagePool;
    
    shared_ptr<DNNGraphSettings> context;
    
    vector<State<ParallelDataNeuralNode>*> state;
    
    DataSetType dataSetType;
    
    DropoutStrategy* dropout = NULL;
    
    // sorted edges
    vector<Edge*> incomingForwardEdges;
    vector<Edge*> incomingBackwardEdges;
    vector<Edge*> outgoingBackwardEdges;
    vector<Edge*> outgoingForwardEdges;
    
    // seen counts
    vector<int> forwardSeenCount;
    vector<int> backwardSeenCount;
    
    // node output/activation
    std::vector<Eigen::VectorXf,Eigen::aligned_allocator<Eigen::VectorXf> > activation;
public:
    ParallelDataNeuralNode(shared_ptr<GraphSettings> context): Node(context){   
        dropout = new DropoutNull();
        try{
            this->context = std::static_pointer_cast<DNNGraphSettings>(context);
        } catch (const std::bad_cast& e) {
            std::cout << e.what() << "\n";
        }
        
        forwardSeenCount = vector<int>(this->context->numModels,0);
        backwardSeenCount = vector<int>(this->context->numModels,0);
        state = vector<State<ParallelDataNeuralNode>*>(this->context->numModels,NULL);
        activation = vector<Eigen::VectorXf,Eigen::aligned_allocator<Eigen::VectorXf> >(this->context->numModels);
        for(auto& s: state) s = new ForwardTrainState<ParallelDataNeuralNode>();
    }
    virtual ~ParallelDataNeuralNode(){
        delete dropout;
        for(auto& s: state) delete s;
    }
    virtual string getType()=0;
    
    virtual void setWeights(const vector<float>& w){
        cout << "setWeights not implemented for this node, " << m_id << "\n";
    }

    void setDropoutStrategy(DropoutStrategy* d){
        if(dropout) delete dropout;
        dropout = d;
    }
    
    void setState(State<ParallelDataNeuralNode>* _state){
        delete _state;
//        if(state) delete state;
//        state = _state;
    }
    
    virtual bool readyToSend(){
        assert(!state.empty());
        for(int i = 0; i < state.size(); ++i)
            if(state[i]->readyToSend(this, i)) return true;
        return false;
    }  
    
    // Handle sending of messages and routing for the node
    virtual bool onSend(vector<Message*>& msgs){
        assert(!state.empty());
        for(int i = 0; i < state.size(); ++i){
            if(state[i]->readyToSend(this,i)){
                state[i]->onSend(this, msgs, i);
                return true; // only operate on one state during a timestep 
            }
        }
    }
    
    // Handle message receiving
    virtual void onRecv(ForwardPropagationMessage* msg)=0;
    virtual void onRecv(BackwardPropagationMessage* msg)=0;
    
    virtual bool sendBackwardMsgs(vector<Message*>& msgs, int stateIndex)=0;
    virtual bool sendForwardMsgs(vector<Message*>& msgs, int stateIndex)=0;
    
    virtual bool readyToSendForward(int i){
        if(!dropout->unset()) return dropout->readyToSendForward(forwardSeenCount[i]);
        return (forwardSeenCount[i] == incomingForwardEdges.size()); 
    }
    virtual bool readyToSendBackward(int i){
        if(!dropout->unset()) return dropout->readyToSendBackward(backwardSeenCount[i]);
        return (backwardSeenCount[i] == incomingBackwardEdges.size());
    }
protected:
    template<typename TState>
    void swapState(State<ParallelDataNeuralNode>* _state){
        delete _state;
        _state = new TState(); 
    }
};

#endif /* NEURAL_NODE_H */

