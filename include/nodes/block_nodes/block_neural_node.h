
/* 
 * File:   neural_node.h
 * Author: ryan
 *
 * Created on 01 May 2017, 20:22
 */

#ifndef BLOCK_NEURAL_NODE_H
#define BLOCK_NEURAL_NODE_H

#include "nodes/node.h"
#include "graphs/block_neural_network_settings.h"

#include <Eigen/Dense>
#include <memory>
#include <cassert>

using namespace std;
using namespace Eigen;

class BlockNeuralNode: public Node{
public:
    class InputNode;
    class OutputNode;
    class HiddenNode;
    class SyncNode;
protected:
    shared_ptr<BlockNeuralNetworkSettings> settings;
    
    // Are these useful??
    // int nNodes; 
    int layer;
    
    // sorted edges
    vector<Edge*> incomingForwardEdges;
    vector<Edge*> incomingBackwardEdges;
    vector<Edge*> outgoingBackwardEdges;
    vector<Edge*> outgoingForwardEdges;
        
    // seen counts
    int forwardSeenCount = 0;
    int backwardSeenCount = 0;
    
    // node outputs/activations
    MatrixXf output;
public:
    
    BlockNeuralNode(shared_ptr<GraphSettings> context): Node(context){        
        try{
            settings = std::static_pointer_cast<BlockNeuralNetworkSettings>(context);
        } catch (const std::bad_cast& e) {
            std::cout << e.what() << "\n";
        }
    }
    
    virtual string getType()=0;
    
    virtual bool readyToSend(){
        if(!settings->state) assert(0);
        settings->state->readyToSend(this);
    }  
    
    // Handle sending of messages and routing for the node
    virtual bool onSend(vector<Message*>& msgs){
        settings->state->onSend(this, msgs);
    }
    
    // Handle message receiving
    virtual void onRecv(ForwardPropagationMessage* msg)=0;
    virtual void onRecv(BackwardPropagationMessage* msg)=0;
    
    virtual bool sendBackwardMsgs(vector<Message*>& msgs)=0;
    virtual bool sendForwardMsgs(vector<Message*>& msgs)=0;
    
    virtual bool readyToSendForward(){
        return (forwardSeenCount == incomingForwardEdges.size()); 
    }
    virtual bool readyToSendBackward(){
        return (backwardSeenCount == incomingBackwardEdges.size());
    }
};

#endif /* BLOCK_NEURAL_NODE_H */

