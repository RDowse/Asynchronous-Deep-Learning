/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

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
    void onSend(NeuralNode* n, vector<shared_ptr<Message>>& msgs) override;
    bool readyToSend(NeuralNode* n) override;
};

#endif /* PREDICT_STATE_H */

