
/* 
 * File:   block_neural_node.h
 * Author: ryan
 *
 * Created on 30 April 2017, 19:00
 */

#ifndef BLOCK_NEURAL_NODE_H
#define BLOCK_NEURAL_NODE_H

#include <memory>
#include <cassert>

using namespace std;

class BlockNeuralNode{
protected:
    shared_ptr<DNNGraphSettings> settings;
    
    vector<int> ids; // collection of nodes represented by this node
    
    // sorted edges
//    vector<Edge*> incomingForwardEdges;
//    vector<Edge*> incomingBackwardEdges;
//    vector<Edge*> outgoingBackwardEdges;
//    vector<Edge*> outgoingForwardEdges;
    
    // seen counts
    int forwardSeenCount = 0;
    int backwardSeenCount = 0;
    
    // node output/activation
    float output = 0;
public:
    NeuralNode(shared_ptr<GraphSettings> context): Node(context){        
        try{
            settings = std::static_pointer_cast<DNNGraphSettings>(context);
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
    
    virtual bool readyToSendForward(){}
    virtual bool readyToSendBackward(){}
    
    float getOutput() const{ return output; }
};

#endif /* BLOCK_NEURAL_NODE_H */

