
#include "nodes/sync_node.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"
#include "tools/math.h"

std::string SyncNode::m_type = "Sync";
NodeRegister<SyncNode> SyncNode::m_reg(SyncNode::m_type);

// Begining sync node
bool SyncNode::dispatchForwardMsgs(vector<shared_ptr<Message>>& msgs){
    assert(readyToSend());
    assert(inputEdges.size()== m_dataset->training_images[sampleIndex].size());
    
//    if(!validating){
//        cout << "batchindex: " << batchIndex << ", dataindex: " << sampleIndex 
//            << ", epoch: " << epochCount << "\n";
//    } else {
//        cout << "validationindex: " << sampleIndex << endl;
//    }

    auto& images = validating ? 
        m_dataset->validation_images : m_dataset->training_images;
        
    // send out data samples to input nodes
    for(unsigned i=0; i < inputEdges.size(); i++){
        assert( 0 == inputEdges[i]->msgStatus );
        auto msg = make_shared<ForwardPropagationMessage>();
        msg->value = images[sampleIndex][i];
        inputEdges[i]->msg = msg; 
        inputEdges[i]->msgStatus = 
            static_cast<Edge::MessageStatus>(1 + inputEdges[i]->getDelay()); // How long until it is ready?
    }
    
    // trigger bias nodes
    for(unsigned i=0; i < biasEdges.size(); i++){
        assert( 0 == biasEdges[i]->msgStatus );
        biasEdges[i]->msg = make_shared<ForwardPropagationMessage>();
        biasEdges[i]->msgStatus = 
            static_cast<Edge::MessageStatus>(1 + biasEdges[i]->getDelay());
    }
    
    // reset
    tick = false;
    inputSeenCount = 0;
    outputSeenCount = 0; // for validation looping
}

bool SyncNode::dispatchBackwardMsgs(vector<shared_ptr<Message>>& msgs){
    assert(readyToSend());
    
    vector<int>& labels = validating ? 
        m_dataset->validation_labels : m_dataset->training_labels;
    
    // prepare msgs
    msgs.reserve(outgoingEdges.size());
    for(int i = 0; i < outputEdges.size(); ++i){
        auto msg = make_shared<BackwardPropagationMessage>();
        msg->target = (labels[sampleIndex] == i ? 0.9 : -0.9);
        msgs.push_back(msg);
    }
    
    send(msgs,outputEdges);
    
    // reset
    outputSeenCount = 0;
}

void SyncNode::onRecv(shared_ptr<BackwardPropagationMessage> msg){
    inputSeenCount++;
    
    // update flags and indexes once all data is received
    if(inputSeenCount == inputEdges.size() && !validating){
        // start forwardprop and move onto next sample
        m_graph->op = DNNGraphSettings::Operation::forward;
        m_graph->update = false;
        sampleIndex++;
        batchIndex++;
        
        // Reset batchindex and update weights
        if(batchIndex==m_graph->batchSize){
            batchIndex = 0;
            // trigger weight updates
            m_graph->update = true;
            cout << "UPDATING \n";
        }
        
        // Reset trainingindex and validate
        if(sampleIndex==m_dataset->training_images.size()){
            sampleIndex = 0;
            epochCount++;
            if(!m_dataset->validation_images.empty()) validating = true;
            cout << "EPOCH: " << epochCount << endl;
        }
    }
}

void SyncNode::onRecv(shared_ptr<ForwardPropagationMessage> msg){
    outputSeenCount++;
     
    // store output for error calculation
    if(validating)
        validation_outputs.push(pair<int,float>(msg->src,msg->value));
    else 
        training_outputs.push(pair<int,float>(msg->src,msg->value));    
    
    // switch propagation direction
    if(outputSeenCount == outputEdges.size() && !validating){
        m_graph->op = DNNGraphSettings::Operation::backward;
        calculateError(training_outputs,m_dataset->training_labels,sampleIndex,training_error);
    }
    
    // validation index incrementing
    if(outputSeenCount == outputEdges.size() && validating){
        // calculating error
        calculateError(validation_outputs,m_dataset->validation_labels,sampleIndex,validation_error);
        if(sampleIndex == m_dataset->validation_images.size()-1){
            sampleIndex = 0;
            validating = false;
        } else {
            // update sample
            sampleIndex++;
            m_graph->update = false;
        }
    }
}