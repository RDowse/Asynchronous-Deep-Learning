
#include "nodes/sync_node.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"
#include "tools/math.h"

std::string SyncNode::m_type = "Sync";
NodeRegister<SyncNode> SyncNode::m_reg(SyncNode::m_type);

// Begining sync node
bool SyncNode::onSend(shared_ptr<ForwardPropagationMessage> msg){
    assert(readyToSend());
    assert(inputEdges.size()== m_dataset->training_images[m_graph->sample].size());
    
    // get next sample. For training.
    if(m_graph->cmd == DNNGraphSettings::Command::train)
        m_graph->sample = m_graph->sample++ % m_dataset->training_images.size();
    
    vector<shared_ptr<ForwardPropagationMessage>> msgs;
    msgs.reserve(inputEdges.size());
    
    for(unsigned i=0; i < inputEdges.size(); i++){
        msgs.push_back(make_shared<ForwardPropagationMessage>());
        assert( 0 == outgoingEdges[i]->msgStatus );
        
        msgs[i]->value = m_dataset->training_images[m_graph->sample][i];
        inputEdges[i]->msg = msgs[i]; // Copy message into channel
        inputEdges[i]->msgStatus = 
            static_cast<Edge::MessageStatus>(1 + inputEdges[i]->getDelay()); // How long until it is ready?
    }
    tick = false;
    seenCount = 0;
}

void SyncNode::onRecv(shared_ptr<BackwardPropagationMessage> msg){
    seenCount++;
}

// End sync node
bool SyncNode::onSend(shared_ptr<BackwardPropagationMessage> msg){
    assert(readyToSend());
    for(int i=0; i < outputEdges.size(); ++i){
        //outputEdges[i]->msg = msg;
    }
    seenCount = 0;
}

void SyncNode::onRecv(shared_ptr<ForwardPropagationMessage> msg){
    seenCount++;
}
