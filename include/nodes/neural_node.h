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

#include <vector>
#include <memory>

using namespace std;

class State;
class ForwardPropagationMessage;
class BackwardPropagationMessage;

class NeuralNode: public Node{
protected:
    shared_ptr<DNNGraphSettings> settings;
    
    // sorted edges
    vector<Edge*> incomingForwardEdges;
    vector<Edge*> incomingBackwardEdges;
    vector<Edge*> outgoingBackwardEdges;
    vector<Edge*> outgoingForwardEdges;
    
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
            std::cout << e.what() << std::endl;
        }
    }
    
    virtual string getType()=0;
    
    virtual void setWeights(const vector<float>& w){
        cout << "setWeights not implemented for this node, " << m_id << endl;
    }
    
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
    
    float getOutput() const{ return output; }
};

#endif /* NEURAL_NODE_H */

