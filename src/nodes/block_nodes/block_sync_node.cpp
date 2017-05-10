
#include "nodes/block_nodes/block_sync_node.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"
#include "tools/math.h"
#include "states/backward_train_state.h"
#include "states/forward_train_state.h"

std::string BlockNeuralNode::SyncNode::m_type = "BlockSync";
NodeRegister<BlockNeuralNode::SyncNode> BlockNeuralNode::SyncNode::m_reg(BlockNeuralNode::SyncNode::m_type);

void BlockNeuralNode::SyncNode::addEdge(Edge* e) {
    Node::addEdge(e);
    if(e->src->getId() == m_id){
        if(e->dst->getType() == "BlockOutput"){
            outgoingBackwardEdges.push_back(e);
            dstOutputIndex[e->dst->getId()] = map_index++;
            nOutput++;
        } else if(e->dst->getType() == "BlockInput" ||
             e->dst->getType() == "Bias"){ 
            outgoingForwardEdges.push_back(e);
            nInput++;
        } else {
            cout << "Unknown type " << e->dst->getType() << "\n";
            assert(0);
        }
    } else if(e->dst->getId() == m_id){
        if(e->src->getType() == "BlockOutput"){
            incomingForwardEdges.push_back(e);
        } else if(e->src->getType() == "BlockInput"
                || e->src->getType() == "Bias"){
            incomingBackwardEdges.push_back(e);
        } else {
            cout << "Unknown type " << e->src->getType() << "\n";
            assert(0);
        }
    } 
}
    
bool BlockNeuralNode::SyncNode::sendForwardMsgs(vector<Message*>& msgs){
    assert(readyToSendForward());
    assert(dataset!=NULL);
    
    auto& images = validating ? 
        dataset->validation_images : dataset->training_images;
    
    // TODO: check data size matches the input size
    Logging::log(3, "Sending sample %d", sampleIndex);
    
    // splitting images into blocks
    int imagePos = 0;
    int miniBatchSize = 1;
    
    assert(settings->blockTopology[0].size() == outgoingForwardEdges.size());
    
    // send out data samples to input nodes
    for(unsigned i = 0; i < outgoingForwardEdges.size(); i++){
        assert(outgoingForwardEdges[i]->dst->getType() == "BlockInput");
        assert( 0 == outgoingForwardEdges[i]->msgStatus );
        // standard msg prep
        auto msg = new ForwardPropagationMessage();
        msg->src = m_id;
        msg->dst = outgoingForwardEdges[i]->dst->getId();
        
        // get the size of the image block
        int blockSize = settings->blockTopology[0][i];
        // copy block of image to msg
        msg->matActivation = images.block(imagePos,trainingIndices[sampleIndex],
                blockSize,miniBatchSize);
        // index pos to the next image block
        imagePos += settings->blockTopology[0][i];
        msgs.push_back(msg);
    }
    
    // reset
    tick = false;
    backwardSeenCount = 0; 
}

bool BlockNeuralNode::SyncNode::sendBackwardMsgs(vector<Message*>& msgs){
    assert(readyToSendBackward());

    auto& labels = validating ? 
        dataset->validation_labels : dataset->training_labels;
    
    Logging::log(3, "Sending sample %d backward", trainingIndices[sampleIndex]);
    // prepare msgs
    int outputSize = settings->netTopology.back();
    MatrixXf target = MatrixXf::Constant(outputSize,settings->miniBatchSize,actMin);
    target(labels[trainingIndices[sampleIndex]],0) = actMax; // TODO: correct for batchsize

    msgs.reserve(outgoingBackwardEdges.size());
    for(int i = 0; i < outgoingBackwardEdges.size(); ++i){
        auto msg = new BackwardPropagationMessage();
        msg->src = m_id;
        msg->dst = outgoingBackwardEdges[i]->dst->getId();
        
        assert(outgoingBackwardEdges[i]->dst->getType() == "BlockOutput");
        int blockSize = settings->blockTopology.back().front();
        msg->matTarget = target.block(i*blockSize,0,blockSize,target.cols());
        
        msgs.push_back(msg);
        cout << "Sending sample" << trainingIndices[sampleIndex] << " " << msg->target << "\n";
    }
    
    // reset
    forwardSeenCount = 0;
}

bool BlockNeuralNode::SyncNode::readyToSendForward(){
    return ((backwardSeenCount == incomingBackwardEdges.size()) && epochCount<=settings->maxEpoch) || tick; 
}
bool BlockNeuralNode::SyncNode::readyToSendBackward(){
    return (forwardSeenCount == incomingForwardEdges.size());
}

void BlockNeuralNode::SyncNode::onRecv(BackwardPropagationMessage* msg){
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
        if(sampleIndex==dataset->training_images.cols()){
            // calulate training error for current epoch
//            float sum = accumulate(error.begin(),error.end(),0.0);
//            if(sum <= 0.01){
//                cout << "final error" << sum << "\n";
//                exit(0);
//            }
            // allow for sampling without replacement
            std::random_shuffle(trainingIndices.begin(), trainingIndices.end());
            sampleIndex = 0;
            epochCount++;
            cout << "EPOCH: " << epochCount << "\n" << "\n";
        }
    }
}

void BlockNeuralNode::SyncNode::onRecv(ForwardPropagationMessage* msg){
    forwardSeenCount++;    

    if(!output.size()) initOutput();
    int index = dstOutputIndex[msg->src];
    int blockSize = settings->blockTopology.back()[0];
    output.block(index*blockSize,0,blockSize,output.cols()) += msg->matActivation;
    
//    float target = (dataset->training_labels(trainingIndices[sampleIndex]) == index ? actMax : actMin);
//    training_error += 0.5*pow((target-msg->activation),2);
    
    delete msg;
    
    // switch propagation direction
    if(readyToSendBackward() && !validating){
        assert(trainingIndices[sampleIndex] < min_error.size());
        assert(trainingIndices[sampleIndex] < error.size());
//        min_error[trainingIndices[sampleIndex]] = min(training_error,min_error[trainingIndices[sampleIndex]]);
//        error[trainingIndices[sampleIndex]] = training_error;
//        
//        cout << "Training error sample" << trainingIndices[sampleIndex] << " " << training_error << "\n";
//        cout << "Minimum error sample" << trainingIndices[sampleIndex] << ": " << min_error[trainingIndices[sampleIndex]] << "\n";
        
        cout << "\n";
        cout << "Epoch "<< epochCount << " (targ) " << dataset->training_labels(sampleIndex) << "\n";
        cout << output << "\n\n";
        
        training_error = 0;

        delete settings->state;
        settings->state = new BackwardTrainState();
    }
}
