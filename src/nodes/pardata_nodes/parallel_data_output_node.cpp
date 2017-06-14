
#include "nodes/pardata_nodes/parallel_data_output_node.h"

#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"
#include "tools/math.h"

std::string ParallelDataNeuralNode::OutputNode::m_type = "Output";

void ParallelDataNeuralNode::OutputNode::addEdge(Edge* e) {
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

bool ParallelDataNeuralNode::OutputNode::sendForwardMsgs(vector<Message*>& msgs, int stateIndex){
    assert(readyToSendForward(stateIndex));
    
    // calulate output activation
    activation[stateIndex] = input[stateIndex].unaryExpr(context->activationFnc);
    
    msgs.reserve(outgoingForwardEdges.size());
    for(unsigned i = 0; i < outgoingForwardEdges.size(); i++){
        assert( 0 == outgoingForwardEdges[i]->msgStatus );
        auto msg = forwardMessagePool->getMessage();
        msg->src = m_id;
        msg->dst = outgoingForwardEdges[i]->dst->getId();
        msg->batchNum = batchNum;
        msg->batchIndex = stateIndex;
        
        msg->activation = activation[stateIndex];
        msgs.push_back(msg);
    }
    
    // reset state
    input[stateIndex].setZero(input[stateIndex].size());
    forwardSeenCount[stateIndex] = 0;
    
    delete state[stateIndex];
    state[stateIndex] = new BackwardTrainState<ParallelDataNeuralNode>();
}

bool ParallelDataNeuralNode::OutputNode::sendBackwardMsgs(vector<Message*>& msgs, int stateIndex){
    assert(readyToSendBackward(stateIndex));   
    
    msgs.reserve(outgoingBackwardEdges.size());
    Eigen::VectorXf diff = -(target[stateIndex]-activation[stateIndex]);
    Eigen::VectorXf delta = diff.array() * activation[stateIndex].unaryExpr(context->deltaActivationFnc).array();
    for(unsigned i = 0; i < outgoingBackwardEdges.size(); i++){
        if(dropout->isPrevLayerNodeActive(i)){
            assert( 0 == outgoingBackwardEdges[i]->msgStatus );
            auto msg = backwardMessagePool->getMessage();
            msg->src = m_id; 
            msg->dst = outgoingBackwardEdges[i]->dst->getId();
            msg->batchNum = batchNum;
            msg->batchIndex = stateIndex;

            msg->delta = delta; 
            msgs.push_back(msg);
        }
    }
    
    backwardSeenCount[stateIndex] = 0;
    
    delete state[stateIndex];
    state[stateIndex] = new ForwardTrainState<ParallelDataNeuralNode>();
}

void ParallelDataNeuralNode::OutputNode::onRecv(ForwardPropagationMessage* msg) {
    if(input[msg->batchIndex].size() != msg->activation.size()) 
        input[msg->batchIndex] = Eigen::VectorXf::Zero(msg->activation.size());
    input[msg->batchIndex] += msg->activation;
    forwardSeenCount[msg->batchIndex]++;    
    
    dataSetType = msg->dataSetType;
    
    if(dataSetType==DataSetType::training) dropout->setEnabled(true);
    else dropout->setEnabled(false);
    
    if(!dropout->unset() && msg->batchNum > batchNum){
        dropout->nextStep(msg->batchNum);
        batchNum = msg->batchNum;
    }
    
    forwardMessagePool->returnMessage(msg);
}

void ParallelDataNeuralNode::OutputNode::onRecv(BackwardPropagationMessage* msg) {
    target[msg->batchIndex] = msg->target;
    backwardSeenCount[msg->batchIndex]++;
    
    backwardMessagePool->returnMessage(msg);
}
