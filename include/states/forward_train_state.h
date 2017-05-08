
/* 
 * File:   forward_train_state.h
 * Author: ryan
 *
 * Created on 24 April 2017, 13:19
 */

#ifndef FORWARD_TRAIN_STATE_H
#define FORWARD_TRAIN_STATE_H

#include "states/state.h"

class ForwardTrainState: public State{
    public:  
    void onSend(NeuralNode* n, vector<Message*>& msgs) override;
    bool readyToSend(NeuralNode* n) override;
    
    void onSend(BlockNeuralNode* n, vector<Message*>& msgs) override;  
    bool readyToSend(BlockNeuralNode* n) override;
};

#endif /* FORWARD_TRAIN_STATE_H */

