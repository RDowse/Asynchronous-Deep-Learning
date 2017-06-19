/* 
 * File:   neural_node.h
 * Author: ryan
 *
 * Created on 24 April 2017, 00:27
 */

#ifndef NEURAL_NODE_H
#define NEURAL_NODE_H

#include "nodes/node.h"
#include "graphs/dnn_graph_settings.h"
#include "misc/message_pool.h"

#include "training/dropout_strategy.h"
#include "training/dropout_null.h"

#include "states/forward_train_state.h"

#include "common.h"

#include <eigen3/Eigen/Dense>
#include <vector>
#include <memory>

using namespace std;

template<typename TNode> class State;
class ForwardPropagationMessage;
class BackwardPropagationMessage;

class NeuralNode: public Node{
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
    
    State<NeuralNode>* state;
    
    DataSetType dataSetType;
    
    DropoutStrategy* dropout = NULL;
    
    // sorted edges
    vector<Edge*> incomingForwardEdges;
    vector<Edge*> incomingBackwardEdges;
    vector<Edge*> outgoingBackwardEdges;
    vector<Edge*> outgoingForwardEdges;
    
    // seen counts
    int forwardSeenCount = 0;
    int backwardSeenCount = 0;
    
    int batchNum = 0;
    
    // node output/activation
    Eigen::VectorXf activation;
public:
    NeuralNode(shared_ptr<GraphSettings> context): Node(context){   
        dropout = new DropoutNull();
        state = new ForwardTrainState<NeuralNode>();
        try{
            this->context = std::static_pointer_cast<DNNGraphSettings>(context);
        } catch (const std::bad_cast& e) {
            std::cout << e.what() << "\n";
        }
    }
    ~NeuralNode(){
        delete dropout;
        delete state;
    }
    
    virtual string getType()=0;
    
    virtual void setWeights(const vector<float>& w){
        cout << "setWeights not implemented for this node, " << id << "\n";
    }
    
    void setState(State<NeuralNode>* _state){
        if(state) delete state;
        state = _state;
    }

    void setDropoutStrategy(DropoutStrategy* d){
        if(dropout) delete dropout;
        dropout = d;
    }
    
    virtual bool readyToSend(){
        assert(state);
        return state->readyToSend(this);
    }  
    
    // Handle sending of messages and routing for the node
    virtual bool onSend(vector<Message*>& msgs){
        assert(state);
        state->onSend(this, msgs);
    }
    
    // Handle message receiving
    virtual void onRecv(ForwardPropagationMessage* msg)=0;
    virtual void onRecv(BackwardPropagationMessage* msg)=0;
    
    virtual bool sendBackwardMsgs(vector<Message*>& msgs)=0;
    virtual bool sendForwardMsgs(vector<Message*>& msgs)=0;
    
    virtual bool readyToSendForward(){
        if(!dropout->unset()) return dropout->readyToSendForward(forwardSeenCount);
        return (forwardSeenCount == incomingForwardEdges.size()); 
    }
    virtual bool readyToSendBackward(){
        if(!dropout->unset()) return dropout->readyToSendBackward(backwardSeenCount);
        return (backwardSeenCount == incomingBackwardEdges.size());
    }
protected:
    template<typename TState>
    void swapState(){
        delete state;
        state = new TState(); 
    }
};

#endif /* NEURAL_NODE_H */

