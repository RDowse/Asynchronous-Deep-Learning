
#include "nodes/async_nodes/async_sync_node.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"
#include "tools/math.h"
#include "states/backward_train_state.h"
#include "states/forward_train_state.h"

std::string AsyncNeuralNode::SyncNode::m_type = "Sync";

void AsyncNeuralNode::SyncNode::addEdge(Edge* e) {
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
    
bool AsyncNeuralNode::SyncNode::sendForwardMsgs(vector<Message*>& msgs){
    MatrixXf* images = NULL;
    switch(dataSetType){
        case DataSetType::training:
            images = &dataset->training_images;
            if( sampleIndex + context->batchSize > dataset->training_labels.size() ) 
                currBatchSize = dataset->training_labels.size() - sampleIndex;
            else
                currBatchSize = context->batchSize;
            break;
        case DataSetType::validating:
            images = &dataset->validation_images;
            currBatchSize = 1000; 
            break;
        case DataSetType::testing:
            images = &dataset->testing_images;
            currBatchSize = 1000;
            break;
        case DataSetType::training_test:
            images = &dataset->training_images;
            currBatchSize = 1000;
            break;
        default:
            assert(0);
    };
    
    Logging::log(3, "Sending sample %d", sampleIndex);
    
    // send out data samples to input nodes
    msgs.reserve(outgoingForwardEdges.size());
    for(unsigned i = 0, j = 0; i < outgoingForwardEdges.size(); i++){
        assert( 0 == outgoingForwardEdges[i]->msgStatus );
        auto msg = forwardMessagePool->getMessage();
        msg->src = m_id;
        msg->dst = outgoingForwardEdges[i]->dst->getId();
        msg->time = time;
        msg->dataSetType = dataSetType;
        
        // sampling strategy
        if(outgoingForwardEdges[i]->dst->getType() == "Input" 
                && j < images->cols()){
            msg->activation = images->block(sampleIndex,j++,currBatchSize,1);
        } else if(outgoingForwardEdges[i]->dst->getType() == "Bias"){
            msg->activation = Eigen::VectorXf::Ones(currBatchSize);
        } 
        msgs.push_back(msg);
        
        numMessagesSent++;
    }
    
    // reset
    receivedOutput.setZero(receivedOutput.rows(),receivedOutput.cols());
    tick = false;
    backwardSeenCount = 0; 
}

bool AsyncNeuralNode::SyncNode::sendBackwardMsgs(vector<Message*>& msgs){
    Logging::log(3, "Sending sample %d backward", sampleIndex);
    
    auto batchLabels = dataset->training_labels.block(sampleIndex,0,currBatchSize,1);
    
    msgs.reserve(outgoingBackwardEdges.size());
    for(int i = 0; i < outgoingBackwardEdges.size(); ++i){
        auto msg = backwardMessagePool->getMessage();
        msg->src = m_id;
        msg->dst = outgoingBackwardEdges[i]->dst->getId();
        msg->time = time;
        
        Eigen::VectorXf target(currBatchSize);
        if(outgoingBackwardEdges.size() == 1){ // binary classification
            for(int j = 0; j < currBatchSize; ++j)
                target(j) = (batchLabels(j) ? context->actMax : context->actMin);
        } else { // multi classification
            for(int j = 0; j < currBatchSize; ++j)
                target(j) = (batchLabels(j) == i ? context->actMax : context->actMin);
        }
        assert(outgoingBackwardEdges[i]->dst->getType() == "Output");
        msg->target = target;
        msgs.push_back(msg);
        
        numMessagesSent++;
    }
    
    // reset
    forwardSeenCount = 0;
}

bool AsyncNeuralNode::SyncNode::readyToSendForward(){
    return backwardSeenCount == incomingBackwardEdges.size() && 
            (context->epoch < context->maxEpoch || dataSetType != DataSetType::training) || tick; 
}

bool AsyncNeuralNode::SyncNode::readyToSendBackward(){
    return (forwardSeenCount == incomingForwardEdges.size());
}

void AsyncNeuralNode::SyncNode::onRecv(BackwardPropagationMessage* msg){
    backwardSeenCount++;
    backwardMessagePool->returnMessage(msg);
    
    // update flags and indexes once all data is received
    if(readyToSendForward()){
        
        // time step
        time++;

        sampleIndex += currBatchSize;
        
        int nTrainingSamples = dataset->training_labels.size();
        
        // end of epoch, all samples in the training set have been passed
        if(sampleIndex==nTrainingSamples){
            dataset->shuffle();
            
            // next epoch
            context->epoch++;
            
            Logging::log(0,"EPOCH: %d\n",context->epoch);            

            dataSetType = DataSetType::training_test;
            
            // reset
            sampleIndex = 0;
            error = 0;
            accuracy = 0;
        } 
        
        // swap state
        swapState<ForwardTrainState<AsyncNeuralNode>>();
    }
}

void AsyncNeuralNode::SyncNode::onRecv(ForwardPropagationMessage* msg){
    forwardSeenCount++;    
    
    if(!receivedOutput.size() || currBatchSize != receivedOutput.rows()) 
        receivedOutput = Eigen::MatrixXf::Zero(currBatchSize,outgoingBackwardEdges.size());
    
    int index = dstOutputIndex[msg->src];
    
    // network output
    receivedOutput.col(index) += msg->activation;
    
    forwardMessagePool->returnMessage(msg);
    
    // switch propagation direction
    if(readyToSendBackward()){ // || (context->time-timer) >= maxTime){
        // training error
        if(outgoingBackwardEdges.size() == 1){ assert(0); }// todo implement binary version
        
        const VectorXi* labels = NULL;
        string text; int numSamples; DataSetType nextDataSet;
        Eigen::VectorXf* accuracy_ref; Eigen::VectorXf* error_ref;
        switch(dataSetType){
            case DataSetType::training:
                // do nothing
                break;
            case DataSetType::training_test:
                text = "TRAINING ACCURACY";
                nextDataSet = DataSetType::validating;
                numSamples = dataset->training_labels.size();
                labels = &dataset->training_labels;
                
                error_ref = &context->error_training;
                accuracy_ref = &context->accuracy_train;
                break;
            case DataSetType::validating:
                text = "VALIDATION ACCURACY";
                nextDataSet = DataSetType::testing;
                numSamples = dataset->validation_labels.size();
                labels = &dataset->validation_labels;
                
                error_ref = &context->error_validation;
                accuracy_ref = &context->accuracy_validation;
                break;
            case DataSetType::testing:
                text = "TESTING ACCURACY"; 
                nextDataSet = DataSetType::training;
                numSamples = dataset->testing_labels.size();
                labels = &dataset->testing_labels;
                
                error_ref = &context->error_testing;
                accuracy_ref = &context->accuracy_testing;
                break;
            default:
                assert(0);
        };

        if(DataSetType::training != dataSetType){
            const VectorXi& tmp_labels = labels->block(sampleIndex,0,currBatchSize,1);
            
            // Calc training error from target information
            Eigen::MatrixXf target = Eigen::MatrixXf::Constant(currBatchSize,outgoingBackwardEdges.size(),context->actMin);
            for(int i = 0; i < currBatchSize; ++i)
                target( i, (int)tmp_labels(i) ) = context->actMax;
            error += math::mse(target,receivedOutput);
            
            // Calc accuracy of current sample
            MatrixXf::Index maxIndex[currBatchSize];
            VectorXf maxVal(currBatchSize);
            for(int i = 0; i < currBatchSize; ++i){
                maxVal(i) = receivedOutput.row(i).maxCoeff( &maxIndex[i] );
                if(maxIndex[i]==tmp_labels(i)) accuracy++;
            }
            
            // next sample
            sampleIndex += currBatchSize;
            
            // trigger forward propagation for next wave
            backwardSeenCount = incomingBackwardEdges.size();
            forwardSeenCount = 0;
            
            // Output accuracy info
            if(sampleIndex==numSamples){
                Logging::log(0,"%s: %f\n\n",text.c_str(),accuracy/numSamples);

                // save accuracy values
                (*accuracy_ref)(context->epoch-1) = accuracy/numSamples;
                (*error_ref)(context->epoch-1) = error/numSamples;
                
                // switch dataset
                dataSetType = nextDataSet;

                // reset
                sampleIndex = 0;
                error = 0;
                accuracy = 0;
            }
        } else {
            // start backpropagation when training
            swapState<BackwardTrainState<AsyncNeuralNode>>();
        }
    }
}