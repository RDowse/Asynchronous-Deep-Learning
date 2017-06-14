
#include "nodes/pardata_nodes/parallel_data_sync_node.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"
#include "tools/math.h"
#include "states/backward_train_state.h"
#include "states/forward_train_state.h"

std::string ParallelDataNeuralNode::SyncNode::m_type = "Sync";

void ParallelDataNeuralNode::SyncNode::addEdge(Edge* e) {
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
    
bool ParallelDataNeuralNode::SyncNode::sendForwardMsgs(vector<Message*>& msgs, int stateIndex){
    assert(readyToSendForward(stateIndex));
    assert(dataset!=NULL);
    
    // time step
    batchNum++;
    
    auto& images = validating ? 
        dataset->validation_images : dataset->training_images;
    
//    if(validating){
//        batchSize = dataset->validation_labels.size(); 
//    }else{
//        // for batch remainder
//        if( sampleIndex + context->batchSize > dataset->training_labels.size() ) 
//            batchSize = dataset->training_labels.size() - sampleIndex;
//        else 
//            batchSize = context->batchSize;
//    }
    
    // simplified
    batchSize = validating ? dataset->validation_labels.size() : context->batchSize;
    
    Logging::log(3, "Sending sample %d", sampleIndex);
    
    // send out data samples to input nodes
    msgs.reserve(outgoingForwardEdges.size());
    for(unsigned i = 0, j = 0; i < outgoingForwardEdges.size(); i++){
        assert( 0 == outgoingForwardEdges[i]->msgStatus );
        auto msg = forwardMessagePool->getMessage();
        msg->src = m_id;
        msg->dst = outgoingForwardEdges[i]->dst->getId();
        msg->batchNum = batchNum;
        msg->dataSetType = validating ? DataSetType::validating : DataSetType::training;
        msg->batchIndex = stateIndex;
        
        // sampling strategy
        if(outgoingForwardEdges[i]->dst->getType() == "Input" 
                && j < images.cols()){
            msg->activation = images.block(sampleIndex,j++,batchSize,1);
        } else if(outgoingForwardEdges[i]->dst->getType() == "Bias"){
            msg->activation = Eigen::VectorXf::Ones(batchSize);
        } else {
            assert(0); // shouldn't occur
        }
        msgs.push_back(msg);
    }
    assert((batchCount % context->numModels) == stateIndex);
    
    // move to next batch
    activeBatchCount++;
    batchCount++;
    assert(activeBatchCount <= context->numModels);
    
    sampleIndex += batchSize;
    
    // reset
    tick = false;
    backwardSeenCount[stateIndex] = 0; 
}

bool ParallelDataNeuralNode::SyncNode::sendBackwardMsgs(vector<Message*>& msgs, int stateIndex){
    assert(readyToSendBackward(stateIndex));
    
    Logging::log(3, "Sending sample %d backward", sampleIndex-(activeBatchCount*batchSize));

    auto batchLabels = dataset->training_labels.block(sampleIndex-(activeBatchCount*batchSize),0,batchSize,1);
    
    msgs.reserve(outgoingBackwardEdges.size());
    for(int i = 0; i < outgoingBackwardEdges.size(); ++i){
        auto msg = backwardMessagePool->getMessage();
        msg->src = m_id;
        msg->dst = outgoingBackwardEdges[i]->dst->getId();
        msg->batchNum = batchNum;
        msg->batchIndex = stateIndex;
        
        Eigen::VectorXf target(batchSize);
        for(int j = 0; j < batchSize; ++j)
            target(j) = (batchLabels(j) == i ? context->actMax : context->actMin);
        assert(outgoingBackwardEdges[i]->dst->getType() == "Output");
        msg->target = target;
        msgs.push_back(msg);
    }
    
    // reset
    forwardSeenCount[stateIndex] = 0;
}

bool ParallelDataNeuralNode::SyncNode::readyToSendForward(int i){
    return  (backwardSeenCount[i] == incomingBackwardEdges.size() 
            || activeBatchCount < context->numModels
            || tick)
            && (batchCount % context->numModels) == i
            && context->epoch < context->maxEpoch 
            && sampleIndex < dataset->training_labels.size() 
            && activeBatchCount < context->numModels;
}

bool ParallelDataNeuralNode::SyncNode::readyToSendBackward(int i){
    return (forwardSeenCount[i] == incomingForwardEdges.size());
}

void ParallelDataNeuralNode::SyncNode::onRecv(BackwardPropagationMessage* msg){
    backwardSeenCount[msg->batchIndex]++;
    backwardMessagePool->returnMessage(msg);
    
    int batchIndex = msg->batchIndex;
    assert(backwardSeenCount[batchIndex] <= incomingBackwardEdges.size());
    
    if(backwardSeenCount[batchIndex] == incomingBackwardEdges.size())
    {   // state update
        // free up active batch count
        activeBatchCount--;
        assert(activeBatchCount >= 0);
        
        int nTrainingSamples = dataset->training_labels.size();
        
        // end of epoch, all samples in the training set have been passed
        if(sampleIndex==nTrainingSamples && activeBatchCount == 0){
            dataset->shuffle();
            
            // next epoch
            context->epoch++;
                        
            context->accuracy = accuracy/nTrainingSamples;
            context->training_error = training_error/nTrainingSamples;
            
            Logging::log(0,"ACCURACY: %f\n",context->accuracy);
            Logging::log(0,"AVERAGE ERROR: %f\n",context->training_error);
            Logging::log(0,"EPOCH: %d\n",context->epoch);       
            
            // reset
            sampleIndex = 0;
            training_error = 0;
            accuracy = 0;
        } 
        
        delete state[batchIndex];
        state[batchIndex] = new ForwardTrainState<ParallelDataNeuralNode>();
    }
}

void ParallelDataNeuralNode::SyncNode::onRecv(ForwardPropagationMessage* msg){
    forwardSeenCount[msg->batchIndex]++;   
    
    if(!receivedOutput[msg->batchIndex].size() || batchSize != receivedOutput[msg->batchIndex].rows()) 
        receivedOutput[msg->batchIndex] = Eigen::MatrixXf(batchSize,outgoingBackwardEdges.size());
    
    int index = dstOutputIndex[msg->src];
    
    // network output
    receivedOutput[msg->batchIndex].col(index) = msg->activation;
    
    forwardMessagePool->returnMessage(msg);
    
    int batchIndex = msg->batchIndex;
    if(readyToSendBackward(batchIndex)){
        // training error
        if(outgoingBackwardEdges.size() == 1){ assert(0); }// todo implement binary version
        
        int batchSampleIndex = sampleIndex - batchSize*activeBatchCount;
        auto batchLabels = !validating ? dataset->training_labels.block(batchSampleIndex,0,batchSize,1):
            dataset->validation_labels; 
        
        Eigen::MatrixXf target = Eigen::MatrixXf::Constant(batchSize,outgoingBackwardEdges.size(),context->actMin);
        for(int i = 0; i < batchSize; ++i)
            target( i, (int)batchLabels(i) ) = context->actMax;
        training_error += math::mse(target,receivedOutput[msg->batchIndex]);
        
        // TODO: separate accuracy predicition for the training set into a separate forward pass
        MatrixXf::Index maxIndex[batchSize];
        VectorXf maxVal(batchSize);
        for(int i = 0; i < batchSize; ++i){ // multi-classification only
            maxVal(i) = receivedOutput[msg->batchIndex].row(i).maxCoeff( &maxIndex[i] );
            if(maxIndex[i]==batchLabels(i)) accuracy++;
        }

        delete state[msg->batchIndex];
        state[msg->batchIndex] = new BackwardTrainState<ParallelDataNeuralNode>();
    }
}