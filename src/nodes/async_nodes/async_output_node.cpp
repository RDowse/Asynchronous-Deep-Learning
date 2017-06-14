
#include "nodes/async_nodes/async_output_node.h"

#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"
#include "tools/math.h"

std::string AsyncNeuralNode::OutputNode::m_type = "Output";

void AsyncNeuralNode::OutputNode::addEdge(Edge* e) {
    // add to original edge sets
    Node::addEdge(e);
    // check edge belongs to this node
    if(e->src->getId() == m_id){
        if(e->dst->getType() == "Hidden" ||
            e->dst->getType() == "Input" ||
            e->dst->getType() == "Bias"){
            outgoingBackwardEdges.push_back(e);
        } else if(e->dst->getType() == "Sync"){
            outgoingForwardEdges.push_back(e);
        } else {
            cout << "Unknown type " << e->dst->getType() << "\n";
            assert(0);
        }
    } else if(e->dst->getId() == m_id){
        if(e->src->getType() == "Hidden" ||
            e->src->getType() == "Input" ||
            e->src->getType() == "Bias"){
            incomingForwardEdges.push_back(e);
        } else if(e->src->getType() == "Sync"){
            incomingBackwardEdges.push_back(e);
        } else {
            cout << "Unknown type " << e->src->getType() << "\n";
            assert(0);
        }
    } 
}

bool AsyncNeuralNode::OutputNode::sendForwardMsgs(vector<Message*>& msgs){
    assert(readyToSendForward());
    
    input *= float(incomingForwardEdges.size()/forwardSeenCount);
    
    // calulate output activation
    activation = input.unaryExpr(context->activationFnc);
    
    msgs.reserve(outgoingForwardEdges.size());
    for(unsigned i = 0; i < outgoingForwardEdges.size(); i++){
        assert( 0 == outgoingForwardEdges[i]->msgStatus );
        auto msg = forwardMessagePool->getMessage();
        msg->src = m_id;
        msg->dst = outgoingForwardEdges[i]->dst->getId();
        msg->batchNum = curr_forward_batch;
        
        msg->activation = activation;
        msgs.push_back(msg);
        
        numMessagesSentForward++;
    }
    
    curr_forward_batch++;
    ready = false;
    
    // reset state
    forwardSeenCount = 0;
    input.setZero(input.size());
    if(dataSetType == DataSetType::training) swapState<BackwardTrainState<AsyncNeuralNode>>();
}

bool AsyncNeuralNode::OutputNode::sendBackwardMsgs(vector<Message*>& msgs){
    assert(readyToSendBackward());   
    
    msgs.reserve(outgoingBackwardEdges.size());
    Eigen::VectorXf diff = -(target-activation);
    Eigen::VectorXf delta = diff.array() * activation.unaryExpr(context->deltaActivationFnc).array();
    for(unsigned i = 0; i < outgoingBackwardEdges.size(); i++){
        if(dropout->isPrevLayerNodeActive(i)){
            assert( 0 == outgoingBackwardEdges[i]->msgStatus );
            auto msg = backwardMessagePool->getMessage();
            msg->src = m_id; 
            msg->dst = outgoingBackwardEdges[i]->dst->getId();
            msg->batchNum = curr_backward_batch;

            msg->delta = delta; 
            msgs.push_back(msg);
            
            numMessagesSentBackward++;
        }
    }
    
    curr_backward_batch++;
    
    backwardSeenCount = 0;
    activation.setZero(activation.rows(),activation.cols());
    swapState<ForwardTrainState<AsyncNeuralNode>>();
}

void AsyncNeuralNode::OutputNode::onRecv(ForwardPropagationMessage* msg) {
    if(input.size() != msg->activation.size()) input = Eigen::VectorXf::Zero(msg->activation.size());
    
    if(forwardDiscardMsgCheck(msg)) return;    
    
    input += msg->activation;
    dataSetType = msg->dataSetType;
    
    if(dataSetType==DataSetType::training) dropout->setEnabled(true);
    else dropout->setEnabled(false);
    
    if(!dropout->unset() && msg->batchNum > batchNum){
        dropout->nextStep(msg->batchNum);
        batchNum = msg->batchNum;
    }
    
    forwardSeenCount++;
    forwardMessagePool->returnMessage(msg);
}

void AsyncNeuralNode::OutputNode::onRecv(BackwardPropagationMessage* msg) {
    target = msg->target;
    backwardSeenCount++;
    
    backwardMessagePool->returnMessage(msg);
}