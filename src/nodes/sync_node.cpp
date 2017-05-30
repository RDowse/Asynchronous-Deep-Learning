
#include "nodes/sync_node.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"
#include "tools/math.h"
#include "states/backward_train_state.h"
#include "states/forward_train_state.h"

std::string NeuralNode::SyncNode::m_type = "Sync";

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
    
    //int batchSize = 0;
    if(validating){
        batchSize = dataset->validation_labels.size(); 
    }else{
        if( sampleIndex + context->batchSize > dataset->training_labels.size() ) 
            batchSize = dataset->training_labels.size() - sampleIndex;
        else
            batchSize = context->batchSize;
    }
    
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
    
    auto batchLabels = dataset->training_labels.block(sampleIndex,0,batchSize,1);
    
    msgs.reserve(outgoingBackwardEdges.size());
    for(int i = 0; i < outgoingBackwardEdges.size(); ++i){
        auto msg = backwardMessagePool->getMessage();
        msg->src = m_id;
        msg->dst = outgoingBackwardEdges[i]->dst->getId();
        msg->time = time;
        
        Eigen::VectorXf target(batchSize);
        if(outgoingBackwardEdges.size() == 1){ // binary classification
            for(int j = 0; j < batchSize; ++j)
                target(j) = (batchLabels(j) ? context->actMax : context->actMin);
        } else { // multi classification
            for(int j = 0; j < batchSize; ++j)
                target(j) = (batchLabels(j) == i ? context->actMax : context->actMin);
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
        
        sampleIndex += batchSize;
        
        int nTrainingSamples = dataset->training_labels.size();
        
        // end of epoch, all samples in the training set have been passed
        if(sampleIndex==nTrainingSamples){
            dataset->shuffle();
            
            // next epoch
            context->epoch++;
                        
            context->accuracy = accuracy/nTrainingSamples;
            context->training_error = training_error/nTrainingSamples;
            
            Logging::log(0,"ACCURACY: %f\n",context->accuracy);
            Logging::log(0,"AVERAGE ERROR: %f\n",context->training_error);
            Logging::log(0,"EPOCH: %d\n",context->epoch);            

            validating = true;
            
            // reset
            sampleIndex = 0;
            training_error = 0;
            accuracy = 0;
        } 
        
        // swap state
        swapState<ForwardTrainState<NeuralNode>>();
    }
}

void NeuralNode::SyncNode::onRecv(ForwardPropagationMessage* msg){
    forwardSeenCount++;    
    
    if(!receivedOutput.size() || batchSize != receivedOutput.rows()) 
        receivedOutput = Eigen::MatrixXf(batchSize,outgoingBackwardEdges.size());
    
    int index = dstOutputIndex[msg->src];
    
    // network output
    receivedOutput.col(index) = msg->activation;
    
    forwardMessagePool->returnMessage(msg);
    
    // switch propagation direction
    if(readyToSendBackward()){
        // training error
        if(outgoingBackwardEdges.size() == 1){ assert(0); }// todo implement binary version
        
        auto batchLabels = !validating ? dataset->training_labels.block(sampleIndex,0,batchSize,1):
            dataset->validation_labels; 
        
        Eigen::MatrixXf target = Eigen::MatrixXf::Constant(batchSize,outgoingBackwardEdges.size(),context->actMin);
        for(int i = 0; i < batchSize; ++i)
            target( i, (int)batchLabels(i) ) = context->actMax;
        training_error += math::mse(target,receivedOutput);
        
        // for accuracy predicition, should really be in a separate forward pass
        MatrixXf::Index maxIndex[batchSize];
        VectorXf maxVal(batchSize);
        for(int i = 0; i < batchSize; ++i){
            if(outgoingBackwardEdges.size() == 1){ // binary classification
                auto mid = (context->actMax - context->actMin) / 2;
                if((receivedOutput(i) > mid)==batchLabels(i)) accuracy++;
            } else { // multi-classification
                maxVal(i) = receivedOutput.row(i).maxCoeff( &maxIndex[i] );
                if(maxIndex[i]==batchLabels(i)) accuracy++;
            }
        }

        // Swap state
        if(!validating) swapState<BackwardTrainState<NeuralNode>>();
        
        if(validating){
            Logging::log(0,"VALIDATION ACCURACY: %f\n\n",accuracy/dataset->validation_labels.size());
            
            // trigger next wave
            validating = false;
            backwardSeenCount = incomingBackwardEdges.size();

            // reset
            forwardSeenCount = 0;
            sampleIndex = 0;
            training_error = 0;
            accuracy = 0;
        }
    }
}