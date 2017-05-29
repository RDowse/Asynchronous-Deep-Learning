
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
    
    // time step
    time++;
    
    auto& images = validating ? 
        dataset->validation_images : dataset->training_images;
    
    int batchSize = validating ? dataset->validation_labels.size() : context->batchSize;
    
    // TODO: check data size matches the input size
    Logging::log(3, "Sending sample %d", sampleIndex);
    
    // send out data samples to input nodes
    msgs.reserve(outgoingForwardEdges.size());
    for(unsigned i = 0, j = 0; i < outgoingForwardEdges.size(); i++){
        assert( 0 == outgoingForwardEdges[i]->msgStatus );
        auto msg = forwardMessagePool->getMessage();
        msg->src = m_id;
        msg->dst = outgoingForwardEdges[i]->dst->getId();
        msg->time = time;
        msg->dataSetType = validating ? DataSetType::validation : DataSetType::training;
        
        // sampling strategy
        if(outgoingForwardEdges[i]->dst->getType() == "Input" 
                && j < images.cols()){
            msg->activation = images.block(sampleIndex,j++,batchSize,1);
        } else if(outgoingForwardEdges[i]->dst->getType() == "Bias"){
            msg->activation = Eigen::VectorXf::Ones(batchSize);
        } 
        msgs.push_back(msg);
    }
    
    // reset
    tick = false;
    backwardSeenCount = 0; 
}

bool NeuralNode::SyncNode::sendBackwardMsgs(vector<Message*>& msgs){
    assert(readyToSendBackward());
    
    Logging::log(3, "Sending sample %d backward", sampleIndex);
    
    float actMax = context->activationFnc(5);
    float actMin = context->activationFnc(-5);
    
    auto batchLabels = dataset->training_labels.block(sampleIndex,0,context->batchSize,1);
    
    msgs.reserve(outgoingBackwardEdges.size());
    for(int i = 0; i < outgoingBackwardEdges.size(); ++i){
        auto msg = backwardMessagePool->getMessage();
        msg->src = m_id;
        msg->dst = outgoingBackwardEdges[i]->dst->getId();
        msg->time = time;
        
        Eigen::VectorXf target(context->batchSize);
        if(outgoingBackwardEdges.size() == 1){ // binary classification
            for(int j = 0; j < context->batchSize; ++j)
                target(j) = (batchLabels(j) ? actMax : actMin);
        } else { // multi classification
            for(int j = 0; j < context->batchSize; ++j)
                target(j) = (batchLabels(j) == i ? actMax : actMin);
        }
        
        assert(outgoingBackwardEdges[i]->dst->getType() == "Output");
        msg->target = target;
        msgs.push_back(msg);
    }
    
    // reset
    forwardSeenCount = 0;
}

bool NeuralNode::SyncNode::readyToSendForward(){
    return (backwardSeenCount == incomingBackwardEdges.size() && context->epoch <= context->maxEpoch) || tick; 
}
bool NeuralNode::SyncNode::readyToSendBackward(){
    return (forwardSeenCount == incomingForwardEdges.size());
}

void NeuralNode::SyncNode::onRecv(BackwardPropagationMessage* msg){
    backwardSeenCount++;
    backwardMessagePool->returnMessage(msg);
    
    // update flags and indexes once all data is received
    if(readyToSendForward()){
        
        sampleIndex+= context->batchSize;
        
        int nTrainingSamples = dataset->training_labels.size();
        
        // end of epoch, all samples in the training set have been passed
        if(sampleIndex==nTrainingSamples){
            dataset->shuffle();
            
            // next epoch
            context->epoch++;
            
            Logging::log(0,"ACCURACY: %f\n",accuracy/nTrainingSamples);
            Logging::log(0,"TOTAL ERROR: %f\n",training_error);
            Logging::log(0,"EPOCH: %d\n",context->epoch);            
            
            context->accuracy = accuracy/nTrainingSamples;
            context->training_error = training_error;
            
            validating = true;
            
            // reset
            sampleIndex = 0;
            training_error = 0;
            accuracy = 0;
        } 
        
        // swap state
        if(!lastState){
            context->state = new ForwardTrainState();
        } else {
            State* tmpState = lastState;
            lastState = context->state;
            context->state = tmpState;
        }
    }
}

void NeuralNode::SyncNode::onRecv(ForwardPropagationMessage* msg){
    forwardSeenCount++;    
    
    int index = dstOutputIndex[msg->src];
    float actMax = context->activationFnc(5);
    float actMin = context->activationFnc(-5);

    int batchSize = !validating ? context->batchSize : dataset->validation_labels.size();
    auto batchLabels = !validating ? dataset->training_labels.block(sampleIndex,0,context->batchSize,1):
        dataset->validation_labels;  
    
    if(!out.size() || batchSize != out.rows()) 
        out = Eigen::MatrixXf(batchSize,outgoingBackwardEdges.size());
    
    Eigen::VectorXf target(msg->activation.size());
    assert(target.size() == batchLabels.size());
    if(outgoingBackwardEdges.size() == 1){ // binary classifications
        for(int i = 0; i < target.size(); ++i)
            target(i) = batchLabels(i) ? actMax : actMin;
    } else {    // multi classification
        for(int i = 0; i < target.size(); ++i)
            target(i) = (batchLabels(i) == index ? actMax : actMin);
    }
    
    // network output
    out.col(index) = msg->activation;
    
    // mse error
    Eigen::VectorXf diff = target - msg->activation;
    auto tmp = float(diff.transpose()*diff); // square
    training_error += 0.5*(tmp);
    
    forwardMessagePool->returnMessage(msg);
    
    // switch propagation direction
    if(readyToSendBackward()){
        // for accuracy predicition, should really be in a separate forward pass
        MatrixXf::Index maxIndex[batchSize];
        VectorXf maxVal(batchSize);
        for(int i=0;i<batchSize;++i){
            if(outgoingBackwardEdges.size() == 1){ // binary classification
                auto mid = (actMax - actMin) / 2;
                if((out(i) > mid)==batchLabels(i)) accuracy++;
            } else { // multi-classification
                maxVal(i) = out.row(i).maxCoeff( &maxIndex[i] );
                if(maxIndex[i]==batchLabels(i)) accuracy++;
            }
        }

        // Swap state
        if(!validating){
            if(!lastState){
                context->state = new BackwardTrainState();
            } else {
                State* tmpState = lastState;
                lastState = context->state;
                context->state = tmpState;
            }
        }
                
        if(validating){
            Logging::log(0,"VALIDATION ACCURACY: %f\n\n",accuracy/dataset->validation_labels.size());
            
            validating = false;
            
            // trigger next wave
            backwardSeenCount = incomingBackwardEdges.size();
            forwardSeenCount = 0;
            
            // reset
            sampleIndex = 0;
            training_error = 0;
            accuracy = 0;
        }
    }
}