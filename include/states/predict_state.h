/* 
 * File:   predict_state.h
 * Author: ryan
 *
 * Created on 24 April 2017, 01:41
 */

#ifndef PREDICT_STATE_H
#define PREDICT_STATE_H

#include "states/state.h"

class PredictState: public State{
    void onSend(NeuralNode* n, vector<Message*>& msgs) override;
    bool readyToSend(NeuralNode* n) override;
};

#endif /* PREDICT_STATE_H */

