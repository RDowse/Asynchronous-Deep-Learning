/* 
 * File:   neural_node.h
 * Author: ryan
 *
 * Created on 24 April 2017, 00:27
 */

#ifndef ASYNC_NEURAL_NODE_H
#define ASYNC_NEURAL_NODE_H

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
#include <climits>

using namespace std;

template<typename TNode> class State;
class ForwardPropagationMessage;
class BackwardPropagationMessage;

class AsyncNeuralNode: public Node{
public:
    class InputNode;
    class HiddenNode;
    class OutputNode;
    class BiasNode;
    class SyncNode;
    
    // message counts
    int discardedForwardMessageCount = 0;
    int discardedBackwardMessageCount = 0;
    int numMessagesSentForward = 0;
    int numMessagesSentBackward = 0;
protected:
    static MessagePool<ForwardPropagationMessage>* forwardMessagePool;
    static MessagePool<BackwardPropagationMessage>* backwardMessagePool;
    
    shared_ptr<DNNGraphSettings> context;
    
    State<AsyncNeuralNode>* state;
    
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
    
    // batch index
    int curr_forward_batch = 0;
    int curr_backward_batch = 0;
    
    bool ready = false;
    
    int batchNum = 0;
    
    // node output/activation
    Eigen::VectorXf activation;
public:
    AsyncNeuralNode(shared_ptr<GraphSettings> context): Node(context){   
        dropout = new DropoutNull();
        state = new ForwardTrainState<AsyncNeuralNode>();
        try{
            this->context = std::static_pointer_cast<DNNGraphSettings>(context);
        } catch (const std::bad_cast& e) {
            std::cout << e.what() << "\n";
        }
    }
    ~AsyncNeuralNode(){
        delete dropout;
        delete state;
    }
    
    virtual string getType()=0;
    
    virtual void setWeights(const vector<float>& w){
        cout << "setWeights not implemented for this node, " << id << "\n";
    }
    
    void setState(State<AsyncNeuralNode>* _state){
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
    
    // Msg timing and coordination
    bool forwardDiscardMsgCheck(ForwardPropagationMessage* msg);
    bool backwardDiscardMsgCheck(BackwardPropagationMessage* msg);
    
    virtual bool readyToSendForward(){
        if(!dropout->unset())return dropout->readyToSendForward(forwardSeenCount) || 
            (ready && dataSetType == DataSetType::training);
        return forwardSeenCount == incomingForwardEdges.size() || 
            (ready && dataSetType == DataSetType::training); 
    }
    virtual bool readyToSendBackward(){
        if(!dropout->unset()) return dropout->readyToSendBackward(backwardSeenCount)|| 
            (ready && dataSetType == DataSetType::training);
        return backwardSeenCount == incomingBackwardEdges.size() || 
            (ready && dataSetType == DataSetType::training);
    }
protected:
    template<typename TState>
    void swapState(){
        delete state;
        state = new TState(); 
    }
};

#endif /* NEURAL_NODE_H */

