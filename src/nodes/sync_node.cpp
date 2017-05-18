
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
    
    auto& images = validating ? 
        dataset->validation_images : dataset->training_images;
    
    // TODO: check data size matches the input size
    Logging::log(3, "Sending sample %d", sampleIndex);
    
    // send out data samples to input nodes
    msgs.reserve(outgoingForwardEdges.size());
    for(unsigned i = 0, j = 0; i < outgoingForwardEdges.size(); i++){
        assert( 0 == outgoingForwardEdges[i]->msgStatus );
        auto msg = forwardMessagePool->getMessage();
        msg->src = m_id;
        msg->dst = outgoingForwardEdges[i]->dst->getId();
        
        // sampling strategy
        if(outgoingForwardEdges[i]->dst->getType() == "Input" 
                && j < images.cols()){
            msg->activation = images.block(sampleIndex,j++,context->batchSize,1);
        } else if(outgoingForwardEdges[i]->dst->getType() == "Bias"){
            msg->activation = Eigen::VectorXf::Ones(context->batchSize);
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
    
    float actMax = context->activationFnc(5);
    float actMin = context->activationFnc(-5);
    
    int currSample = trainingIndices[sampleIndex];
    auto batchLabels = dataset->training_labels.block(currSample,0,context->batchSize,1);
    
    msgs.reserve(outgoingBackwardEdges.size());
    for(int i = 0; i < outgoingBackwardEdges.size(); ++i){
        auto msg = backwardMessagePool->getMessage();
        msg->src = m_id;
        msg->dst = outgoingBackwardEdges[i]->dst->getId();
        
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
    return ((backwardSeenCount == incomingBackwardEdges.size()) && context->epoch <= context->maxEpoch) || tick; 
}
bool NeuralNode::SyncNode::readyToSendBackward(){
    return (forwardSeenCount == incomingForwardEdges.size());
}

void NeuralNode::SyncNode::onRecv(BackwardPropagationMessage* msg){
    backwardSeenCount++;
    
    backwardMessagePool->returnMessage(msg);
    
    // update flags and indexes once all data is received
    if(readyToSendForward()){
        // start forward pass on the next sample
        
        if(!lastState){
            context->state = new ForwardTrainState();
        } else {
            State* tmpState = lastState;
            lastState = context->state;
            context->state = tmpState;
        }
        
        sampleIndex+= context->batchSize; // todo remainder
        
        // end of epoch, all samples in the training set have been passed
        if(sampleIndex==dataset->training_labels.size()){
            if(training_error <= 0.01){
                cout << "final error" << training_error << "\n";
                exit(0);
            }
            
            // allow for sampling without replacement
            auto& images = dataset->training_images;
            auto& labels = dataset->training_labels;
            PermutationMatrix<Dynamic,Dynamic> perm(images.rows());
            perm.setIdentity();
            std::random_shuffle(perm.indices().data(), perm.indices().data()+perm.indices().size());
            images = perm * images; // permute rows
            labels = perm * labels;
            
            // next epoch
            context->epoch++;
            
            Logging::log(0,"ACCURACY: %f\n",accuracy/images.rows());
            Logging::log(0,"TOTAL ERROR: %f\n",training_error);
            Logging::log(0,"EPOCH: %d\n\n",context->epoch);            
            
            // reset
            sampleIndex = 0;
            training_error = 0;
            accuracy = 0;
        }
    }
}

void NeuralNode::SyncNode::onRecv(ForwardPropagationMessage* msg){
    forwardSeenCount++;    
    
    if(!out.size()) out = Eigen::MatrixXf(outgoingBackwardEdges.size(),context->batchSize);
    
    int currSample = trainingIndices[sampleIndex]; // TODO: refactor to member variable
    int index = dstOutputIndex[msg->src];
    
    float actMax = context->activationFnc(5);
    float actMin = context->activationFnc(-5);
    
    // set target (TODO FIX FOR XOR BINARY CLASSIFICATION)
    auto batchLabels = dataset->training_labels.block(currSample,0,context->batchSize,1);    
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
    out.row(index) = msg->activation;
    
    // mse error
    Eigen::VectorXf diff = target - msg->activation;
    auto tmp = float(diff.transpose()*diff);
    training_error += 0.5*(tmp);
    
    forwardMessagePool->returnMessage(msg);
    
    // switch propagation direction
    if(readyToSendBackward()){
        assert(trainingIndices[sampleIndex] < min_error.size());
        assert(trainingIndices[sampleIndex] < error.size());
        min_error[currSample] = min(training_error,min_error[currSample]);
        
        Logging::log(1,"Training error sample%d %f\n",currSample,training_error);
        Logging::log(1,"Minimum error sample%d %f\n",currSample,min_error[currSample]);
        Logging::log(1,"Epoch %d (targ) %d\n",context->epoch,dataset->training_labels(currSample));
        
        // for accuracy predicition, should really be in a separate forward pass
        MatrixXf::Index maxIndex[context->batchSize];
        VectorXf maxVal(context->batchSize);
        for(int i=0;i<context->batchSize;++i){
            maxVal(i) = out.col(i).maxCoeff( &maxIndex[i] );
            if(maxIndex[i]==batchLabels(i)) accuracy += 1;
        }
        
        if(!lastState){
            context->state = new BackwardTrainState();
        } else {
            State* tmpState = lastState;
            lastState = context->state;
            context->state = tmpState;
        }
    }
}