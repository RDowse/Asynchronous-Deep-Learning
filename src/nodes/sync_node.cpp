
#include "nodes/sync_node.h"
#include "messages/forward_propagation_message.h"
#include "messages/backward_propagation_message.h"
#include "tools/math.h"

std::string SyncNode::m_type = "Sync";
NodeRegister<SyncNode> SyncNode::m_reg(SyncNode::m_type);

// Begining sync node
bool SyncNode::dispatchForwardMsgs(){
    assert(readyToSend());
    assert(inputEdges.size()== m_dataset->training_images[trainingIndex].size());
    
    if(!validating){
        cout << "batchindex: " << batchCount << ", dataindex: " << trainingIndex 
            << ", epoch: " << epochCount << "\n";
    } else {
        cout << "validationindex: " << trainingIndex << endl;
    }
    
    vector< vector<int> >& images = validating ? 
        m_dataset->validation_images : m_dataset->training_images;
        
    // send out data samples to input nodes
    for(unsigned i=0; i < inputEdges.size(); i++){
        assert( 0 == inputEdges[i]->msgStatus );
        auto msg = make_shared<ForwardPropagationMessage>();
        msg->value = images[trainingIndex][i];
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

    // get next sample, for training
    if(m_graph->cmd == DNNGraphSettings::Command::train)
        trainingIndex = trainingIndex++ % images.size();
    
    // reset
    tick = false;
    inputSeenCount = 0;
    outputSeenCount = 0; // for validation looping
}

bool SyncNode::dispatchBackwardMsgs(){
    assert(readyToSend());
    
    // send msg to output to trigger backpropagation
    for(int i=0; i < outputEdges.size(); ++i){
        outputEdges[i]->msg = make_shared<BackwardPropagationMessage>();
        outputEdges[i]->msgStatus = 
            static_cast<Edge::MessageStatus>(1 + outputEdges[i]->getDelay());
    }
    
    // reset
    outputSeenCount = 0;
}

void SyncNode::onRecv(shared_ptr<BackwardPropagationMessage> msg){
    inputSeenCount++;
    // update the batch and epoch once all data is received
    if(inputSeenCount == inputEdges.size() && !validating){
        m_graph->op = DNNGraphSettings::Operation::forward;
        m_graph->update = false;
        if(trainingIndex==m_dataset->training_images.size()-1){
            trainingIndex = 0;
            epochCount++;
            validating = true;
        } else {
            trainingIndex++;
            batchCount++;
            if(batchCount>=m_graph->batchSize){
                batchCount = 0;
                // trigger weight updates
                m_graph->update = true;
                cout << "UPDATING \n";
            }
        }
    }
}

void SyncNode::onRecv(shared_ptr<ForwardPropagationMessage> msg){
    outputSeenCount++;
     
    // store output for error calculation
    if(validating) output.push(pair<int,float>(msg->src,msg->value));
    
    // switch propagation direction
    if(outputSeenCount == outputEdges.size() && !validating) 
        m_graph->op = DNNGraphSettings::Operation::backward;
    
    // validation index incrementing
    if(outputSeenCount == outputEdges.size() && validating){
        
        // calculating error
        vector<pair<int,float>> out;
        while(!output.empty()){
            auto tmp = output.top();
            out.push_back(tmp);
            output.pop();
        }
        sort(out.begin(),out.end(),[](const pair<int,float> &left, const pair<int,float> &right) {
            return left.first < right.first;
        });
        float mse = 0;
        for(int i = 0; i < out.size(); ++i){
            if(i == m_dataset->validation_labels[trainingIndex]-1)
                mse += pow((1 - out[i].second),2);
            else
                mse += pow((-1 - out[i].second),2);
        }
        validation_error += mse;
                
        // update sample
        trainingIndex++;
        m_graph->update = false;
        
        // stop validation 
        if(trainingIndex == m_dataset->validation_images.size()){
            trainingIndex = 0;
            validating = false;
            
            // output validation error
            validation_error/=m_dataset->validation_images.size();
            cout << "VALIDATION ERROR: " << validation_error << endl;
            validation_error = 0;
        } 
    }
}