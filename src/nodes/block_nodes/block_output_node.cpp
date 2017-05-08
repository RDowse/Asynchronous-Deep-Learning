
#include "nodes/block_nodes/block_output_node.h"

#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"
#include "tools/math.h"

std::string BlockNeuralNode::OutputNode::m_type = "BlockOutput";
NodeRegister<BlockNeuralNode::OutputNode> BlockNeuralNode::OutputNode::m_reg(BlockNeuralNode::OutputNode::m_type);

void BlockNeuralNode::OutputNode::addEdge(Edge* e) {
    // add to original edge sets
    Node::addEdge(e);
    // check edge belongs to this node
    if(e->src->getId() == m_id){
        if(e->dst->getType() == "BlockHidden" ||
            e->dst->getType() == "BlockInput" ||
            e->dst->getType() == "Bias"){
            outgoingBackwardEdges.push_back(e);
        } else if(e->dst->getType() == "BlockSync"){
            outgoingForwardEdges.push_back(e);
        } else {
            cout << "Unknown type " << e->dst->getType() << "\n";
            assert(0);
        }
    } else if(e->dst->getId() == m_id){
        if(e->src->getType() == "BlockHidden" ||
            e->src->getType() == "BlockInput" ||
            e->src->getType() == "Bias"){
            incomingForwardEdges.push_back(e);
        } else if(e->src->getType() == "BlockSync"){
            incomingBackwardEdges.push_back(e);
        } else {
            cout << "Unknown type " << e->src->getType() << "\n";
            assert(0);
        }
    } 
}

bool BlockNeuralNode::OutputNode::sendForwardMsgs(vector<Message*>& msgs){
    assert(readyToSendForward());
    
    // sigmoid calculations
    for(int col = 0; col < output.cols(); ++col)
        for(int row = 0; row < output.rows(); ++row)
            output(row,col) = settings->activationFnc(output(row,col));
    
    for(unsigned i = 0; i < outgoingForwardEdges.size(); i++){
        assert( 0 == outgoingForwardEdges[i]->msgStatus );
        auto msg = new ForwardPropagationMessage();
        msg->src = m_id;
        msg->dst = outgoingForwardEdges[i]->dst->getId();
        
        // separate output into blocks
        msg->matActivation = output;
        msgs.push_back(msg);
    }
    
    // reset
    output.Zero(output.rows(),output.cols());
    forwardSeenCount = 0;
}

bool BlockNeuralNode::OutputNode::sendBackwardMsgs(vector<Message*>& msgs){
    assert(readyToSendBackward());   
        
    // sigmoid calculations
    MatrixXf deltaAct(output.rows(),output.cols());
    for(int col = 0; col < output.cols(); ++col)
        for(int row = 0; row < output.rows(); ++row)
            deltaAct(row,col) = settings->deltaActivationFnc(output(row,col));
    
    MatrixXf delta = (target-output)*deltaAct;
//    for(unsigned i = 0; i < outgoingBackwardEdges.size(); i++){
//        assert( 0 == outgoingBackwardEdges[i]->msgStatus );
//        auto msg = new BackwardPropagationMessage();
//        msg->src = m_id;
//        msg->dst = outgoingBackwardEdges[i]->dst->getId();
//        msg->delta = delta; 
//        msgs.push_back(msg);
//    }
    
    backwardSeenCount = 0;
}

void BlockNeuralNode::OutputNode::onRecv(ForwardPropagationMessage* msg) {
    if(!output.size()) initOutput();
    output += msg->matActivation;
    forwardSeenCount++;
    
    delete msg;
}

void BlockNeuralNode::OutputNode::onRecv(BackwardPropagationMessage* msg) {
    if(!target.size()) initTarget();
    target = msg->matTarget;
    backwardSeenCount++;
    
    delete msg;
}