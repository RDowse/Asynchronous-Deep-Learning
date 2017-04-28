
#include "nodes/sync_node.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"
#include "tools/math.h"
#include "states/backward_train_state.h"
#include "states/forward_train_state.h"

std::string SyncNode::m_type = "Sync";
NodeRegister<SyncNode> SyncNode::m_reg(SyncNode::m_type);

void SyncNode::addEdge(Edge* e) {
    Node::addEdge(e);
    if(e->src->getId() == m_id){
        if(e->dst->getType() == "Output"){
            outgoingBackwardEdges.push_back(e);
            dstOutputIndex[e->dst->getId()] = map_index++;
            // refactor
            out.push_back(0);
        } else if(e->dst->getType() == "Input" ||
             e->dst->getType() == "Bias"){ 
            outgoingForwardEdges.push_back(e);
        } else {
            cout << "Unknown type " << e->dst->getType() << endl;
            assert(0);
        }
    } else if(e->dst->getId() == m_id){
        if(e->src->getType() == "Output"){
            incomingForwardEdges.push_back(e);
        } else if(e->src->getType() == "Input"
                || e->src->getType() == "Bias"){
            incomingBackwardEdges.push_back(e);
        } else {
            cout << "Unknown type " << e->src->getType() << endl;
            assert(0);
        }
    } 
}
    
bool SyncNode::sendForwardMsgs(vector<Message*>& msgs){
    assert(readyToSendForward());
    assert(dataset!=NULL);
    
    auto& images = validating ? 
        dataset->validation_images : dataset->training_images;
    
    // TODO: check data size matches the input size
    Logging::log(3, "Sending sample %d", sampleIndex);
    
    // send out data samples to input nodes
    for(unsigned i = 0, j = 0; i < outgoingForwardEdges.size(); i++){
        assert( 0 == outgoingForwardEdges[i]->msgStatus );
        auto msg = new ForwardPropagationMessage();
        msg->src = m_id;
        msg->dst = outgoingForwardEdges[i]->dst->getId();
        // sampling strategy
        if(outgoingForwardEdges[i]->dst->getType() == "Input" 
                && j < images[trainingIndices[sampleIndex]].size()){
            msg->value = images[trainingIndices[sampleIndex]][j++];
        } else if(outgoingForwardEdges[i]->dst->getType() == "Bias"){
            msg->value = 0;
        } 
        msgs.push_back(msg);
    }
    
    send(msgs,outgoingForwardEdges);
    
    // reset
    tick = false;
    backwardSeenCount = 0; 
}

bool SyncNode::sendBackwardMsgs(vector<Message*>& msgs){
    assert(readyToSendBackward());

    vector<int>& labels = validating ? 
        dataset->validation_labels : dataset->training_labels;
    
    // TODO: check data size matches the input size
    Logging::log(3, "Sending sample %d backward", trainingIndices[sampleIndex]);
    // prepare msgs
    msgs.reserve(outgoingBackwardEdges.size());
    for(int i = 0; i < outgoingBackwardEdges.size(); ++i){
        auto msg = new BackwardPropagationMessage();
        msg->src = m_id;
        msg->dst = outgoingBackwardEdges[i]->dst->getId();
        if(outgoingBackwardEdges[i]->dst->getType() == "Output")
            if(outgoingBackwardEdges.size() == 1) // binary classification
                msg->target = (labels[trainingIndices[sampleIndex]] ? actMax : actMin);
            else // multiclassification
                msg->target = (labels[trainingIndices[sampleIndex]] == i ? actMax : actMin);
        else 
            assert(0); // should not occur
        msgs.push_back(msg);
        //cout << "Sending sample" << trainingIndices[sampleIndex] << " " << msg->target << endl;
    }
    
    send(msgs,outgoingBackwardEdges);
    
    // reset
    forwardSeenCount = 0;
}

bool SyncNode::readyToSendForward(){
    return ((backwardSeenCount == incomingBackwardEdges.size()) && epochCount<=settings->maxEpoch) || tick; 
}
bool SyncNode::readyToSendBackward(){
    return (forwardSeenCount == incomingForwardEdges.size());
}

void SyncNode::onRecv(BackwardPropagationMessage* msg){
    backwardSeenCount++;
    
    delete msg;
    
    // update flags and indexes once all data is received
    if(readyToSendForward() && !validating){
        // start forward pass on the next sample
        delete settings->state;
        settings->state = new ForwardTrainState();
        settings->update = true;
        sampleIndex++;
        
        // end of epoch, all samples in the training set have been passed
        if(sampleIndex==dataset->training_images.size()){
            // calulate training error for current epoch
            float sum = accumulate(error.begin(),error.end(),0.0);
            if(sum <= 0.01){
                cout << "final error" << sum << endl;
                exit(0);
            }
            // allow for sampling without replacement
            std::random_shuffle(trainingIndices.begin(), trainingIndices.end());
            sampleIndex = 0;
            epochCount++;
            cout << "EPOCH: " << epochCount << endl << endl;
        }
    }
}

void SyncNode::onRecv(ForwardPropagationMessage* msg){
    forwardSeenCount++;    
    
    int index = dstOutputIndex[msg->src];
    float target = (dataset->training_labels[trainingIndices[sampleIndex]] == index ? actMax : actMin);
    training_error += 0.5*pow((target-msg->value),2);
    
    out[index] = msg->value;
    
    delete msg;
    
    // switch propagation direction
    if(readyToSendBackward() && !validating){
        assert(trainingIndices[sampleIndex] < min_error.size());
        assert(trainingIndices[sampleIndex] < error.size());
        min_error[trainingIndices[sampleIndex]] = min(training_error,min_error[trainingIndices[sampleIndex]]);
        error[trainingIndices[sampleIndex]] = training_error;
        
        cout << "Training error sample" << trainingIndices[sampleIndex] << " " << training_error << endl;
        cout << "Minimum error sample" << trainingIndices[sampleIndex] << ": " << min_error[trainingIndices[sampleIndex]] << endl;
        
        cout << "\n";
        cout << "Epoch "<< epochCount << " (targ) " << dataset->training_labels[sampleIndex] << "\n";
        for(auto i: out)
            cout << i << " ";
        cout << "\n\n";
        
        training_error = 0;

        delete settings->state;
        settings->state = new BackwardTrainState();
    }
}