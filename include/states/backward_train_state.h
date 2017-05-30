/* 
 * File:   backward_train_state.h
 * Author: ryan
 *
 * Created on 24 April 2017, 13:20
 */

#ifndef BACKWARD_TRAIN_STATE_H
#define BACKWARD_TRAIN_STATE_H

#include "states/state.h"

template<typename TNode>
class BackwardTrainState: public State<TNode>{
    public:
    void onSend(TNode* n, vector<Message*>& msgs) override;
    bool readyToSend(TNode* n) override;
};

#endif /* BACKWARD_TRAIN_STATE_H */

