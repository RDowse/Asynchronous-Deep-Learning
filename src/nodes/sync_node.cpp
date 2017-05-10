
#include "nodes/sync_node.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"
#include "tools/math.h"
#include "states/backward_train_state.h"
#include "states/forward_train_state.h"

std::string NeuralNode::SyncNode::m_type = "Sync";
NodeRegister<NeuralNode::SyncNode> NeuralNode::SyncNode::m_reg(NeuralNode::SyncNode::m_type);

void NeuralNode::SyncNode::addEdge(Edge* e) {
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
            cout << "Unknown type " << e->dst->getType() << "\n";
            assert(0);
        }
    } else if(e->dst->getId() == m_id){
        if(e->src->getType() == "Output"){
            incomingForwardEdges.push_back(e);
        } else if(e->src->getType() == "Input"
                || e->src->getType() == "Bias"){
            incomingBackwardEdges.push_back(e);
        } else {
            cout << "Unknown type " << e->src->getType() << "\n";
            assert(0);
        }
    } 
}
    
bool NeuralNode::SyncNode::sendForwardMsgs(vector<Message*>& msgs){
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
                && j < images.rows()){
            msg->activation = images(j++,trainingIndices[sampleIndex]);
        } else if(outgoingForwardEdges[i]->dst->getType() == "Bias"){
            msg->activation = 0;
        } 
        msgs.push_back(msg);
    }
    
    // reset
    tick = false;
    backwardSeenCount = 0; 
}

bool NeuralNode::SyncNode::sendBackwardMsgs(vector<Message*>& msgs){
    assert(readyToSendBackward());

    auto& labels = validating ? 
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
                msg->target = (labels(trainingIndices[sampleIndex]) ? actMax : actMin);
            else // multiclassification
                msg->target = (labels(trainingIndices[sampleIndex]) == i ? actMax : actMin);
        else 
            assert(0); // should not occur
        msgs.push_back(msg);
    }
    
    // reset
    forwardSeenCount = 0;
}

bool NeuralNode::SyncNode::readyToSendForward(){
    return ((backwardSeenCount == incomingBackwardEdges.size()) && epochCount<=settings->maxEpoch) || tick; 
}
bool NeuralNode::SyncNode::readyToSendBackward(){
    return (forwardSeenCount == incomingForwardEdges.size());
}

void NeuralNode::SyncNode::onRecv(BackwardPropagationMessage* msg){
    backwardSeenCount++;
    
    delete msg;
    
    // update flags and indexes once all data is received
    if(readyToSendForward()){
        // start forward pass on the next sample
        delete settings->state;
        settings->state = new ForwardTrainState();
        sampleIndex++;
       
        // end of epoch, all samples in the training set have been passed
        if(sampleIndex==dataset->training_labels.size()){
            // calulate training error for current epoch
            float sum = accumulate(error.begin(),error.end(),0.0);
            if(sum <= 0.01){
                cout << "final error" << sum << "\n";
                exit(0);
            }
            // allow for sampling without replacement
            std::shuffle(std::begin(trainingIndices), std::end(trainingIndices), engine);
            //std::random_shuffle(trainingIndices.begin(), trainingIndices.end());
            sampleIndex = 0;
            epochCount++;
            Logging::log(0,"TOTAL ERROR: %f\n",sum);
            Logging::log(0,"EPOCH: %d\n\n",epochCount);
        }
    }
}

void NeuralNode::SyncNode::onRecv(ForwardPropagationMessage* msg){
    forwardSeenCount++;    
    
    int currSample = trainingIndices[sampleIndex]; // refactor to member variable
    int index = dstOutputIndex[msg->src];
    float target = 0;
    if(outgoingBackwardEdges.size() == 1) {
        target = (dataset->training_labels(currSample) ? actMax : actMin);
    } else {
        target = (dataset->training_labels(currSample) == index ? actMax : actMin);
    }
    training_error += 0.5*pow((target-msg->activation),2);
    out[index] = msg->activation;
    
    delete msg;
    
    // switch propagation direction
    if(readyToSendBackward()){
        assert(trainingIndices[sampleIndex] < min_error.size());
        assert(trainingIndices[sampleIndex] < error.size());
        min_error[currSample] = min(training_error,min_error[currSample]);
        error[currSample] = training_error;
        
        Logging::log(1,"Training error sample%d %f\n",currSample,training_error);
        Logging::log(1,"Minimum error sample%d %f\n",currSample,min_error[currSample]);
        Logging::log(1,"Epoch %d (targ) %d\n",epochCount,dataset->training_labels(currSample));

        stringstream ss;
        for(auto i: out) ss << i << " ";
        Logging::log(1,"Output: %s\n\n",ss.str().c_str()); 
        
        training_error = 0;

        delete settings->state;
        settings->state = new BackwardTrainState();
    }
}