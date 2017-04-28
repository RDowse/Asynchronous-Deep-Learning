
/* 
 * File:   training_strategy.h
 * Author: ryan
 *
 * Created on 25 April 2017, 19:01
 */

#ifndef TRAINING_STRATEGY_H
#define TRAINING_STRATEGY_H

#include "messages/forward_propagation_message.h"
#include "misc/data_wrapper.h"

#include <memory>
#include <vector>

class DNNGraphSettings;

class TrainingStrategy{
protected:
    typedef vector<shared_ptr<ForwardPropagationMessage>>::iterator ForwardMessageIterator;
public:
    TrainingStrategy(){}
    virtual void sample(ForwardMessageIterator first, ForwardMessageIterator last, DataWrapper* dataSet)=0;
    virtual void computeDeltaWeights(   shared_ptr<DNNGraphSettings> context,
                                        float output, 
                                        vector<float>& deltas, 
                                        vector<float>& deltaWeights)=0; 
};

#endif /* TRAINING_STRATEGY_H */

