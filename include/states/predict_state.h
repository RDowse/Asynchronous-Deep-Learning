/* 
 * File:   predict_state.h
 * Author: ryan
 *
 * Created on 24 April 2017, 01:41
 */

#ifndef PREDICT_STATE_H
#define PREDICT_STATE_H

#include "states/state.h"

template<typename TNode>
class PredictState: public State<TNode>{
    public:
    void onSend(TNode* n, vector<Message*>& msgs, int stateIndex) override;
    void onSend(TNode* n, vector<Message*>& msgs) override;
    bool readyToSend(TNode* n) override;
    bool readyToSend(TNode* n, int stateIndex) override;
};

#endif /* PREDICT_STATE_H */

