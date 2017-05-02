
#include "nodes/block_nodes/block_output_node.h"

#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"
#include "tools/math.h"

std::string BlockNode::OutputNode::m_type = "BlockOutput";
NodeRegister<BlockNode::OutputNode> BlockNode::OutputNode::m_reg(BlockNode::OutputNode::m_type);

void BlockNode::OutputNode::addEdge(Edge* e) {
    // add to original edge sets
    Node::addEdge(e);
    // check edge belongs to this node
//    if(e->src->getId() == m_id){
//        if(e->dst->getType() == "Hidden" ||
//            e->dst->getType() == "Input" ||
//            e->dst->getType() == "Bias"){
//            outgoingBackwardEdges.push_back(e);
//        } else if(e->dst->getType() == "Sync"){
//            outgoingForwardEdges.push_back(e);
//        } else {
//            cout << "Unknown type " << e->dst->getType() << "\n";
//            assert(0);
//        }
//    } else if(e->dst->getId() == m_id){
//        if(e->src->getType() == "Hidden" ||
//            e->src->getType() == "Input" ||
//            e->src->getType() == "Bias"){
//            incomingForwardEdges.push_back(e);
//        } else if(e->src->getType() == "Sync"){
//            incomingBackwardEdges.push_back(e);
//        } else {
//            cout << "Unknown type " << e->src->getType() << "\n";
//            assert(0);
//        }
//    } 
}

bool BlockNode::OutputNode::sendForwardMsgs(vector<Message*>& msgs){
    assert(readyToSendForward());
    
    // calulate output activation
    //output = settings->activationFnc(value);
    
    // reset state
//    value = 0;
    forwardSeenCount = 0;
}

bool BlockNode::OutputNode::sendBackwardMsgs(vector<Message*>& msgs){
    assert(readyToSendBackward());   
    
    backwardSeenCount = 0;
}

void BlockNode::OutputNode::onRecv(ForwardPropagationMessage* msg) {
    //value += msg->activation;
    forwardSeenCount++;
    
    delete msg;
}

void BlockNode::OutputNode::onRecv(BackwardPropagationMessage* msg) {
    //target = msg->target;
    backwardSeenCount++;
    
    delete msg;
}