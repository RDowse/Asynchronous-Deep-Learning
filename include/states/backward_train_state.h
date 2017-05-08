/* 
 * File:   backward_train_state.h
 * Author: ryan
 *
 * Created on 24 April 2017, 13:20
 */

#ifndef BACKWARD_TRAIN_STATE_H
#define BACKWARD_TRAIN_STATE_H

#include "states/state.h"

class BackwardTrainState: public State{
    public:
    void onSend(NeuralNode* n, vector<Message*>& msgs) override;
    bool readyToSend(NeuralNode* n) override;
    
    void onSend(BlockNeuralNode* n, vector<Message*>& msgs) override;
    bool readyToSend(BlockNeuralNode* n) override;
};

#endif /* BACKWARD_TRAIN_STATE_H */

