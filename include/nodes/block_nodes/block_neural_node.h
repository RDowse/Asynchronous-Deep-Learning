
/* 
 * File:   neural_node.h
 * Author: ryan
 *
 * Created on 01 May 2017, 20:22
 */

#ifndef BLOCK_NEURAL_NODE_H
#define BLOCK_NEURAL_NODE_H

#include "nodes/node.h"
#include "graphs/dnn_graph_settings.h"

#include <Eigen/Dense>
#include <memory>
#include <cassert>

using namespace std;
using namespace Eigen;

class BlockNeuralNode: public Node{
protected:
    shared_ptr<DNNGraphSettings> settings;
    
    int nNodes;
    
    // collection of nodes represented by this node
    vector<int> ids;
    
    // sorted edges
    vector<Edge*> incomingForwardEdges;
    vector<Edge*> incomingBackwardEdges;
    vector<Edge*> outgoingBackwardEdges;
    vector<Edge*> outgoingForwardEdges;
        
    // seen counts
    int forwardSeenCount = 0;
    int backwardSeenCount = 0;
    
    // node outputs/activations
    VectorXf output;
public:
    
    BlockNeuralNode(shared_ptr<GraphSettings> context): Node(context){        
        try{
            settings = std::static_pointer_cast<DNNGraphSettings>(context);
        } catch (const std::bad_cast& e) {
            std::cout << e.what() << "\n";
        }
    }
    class InputNode;
    class OutputNode;
    class HiddenNode;
    
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

