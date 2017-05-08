
/* 
 * File:   stochastic_momentum_training.h
 * Author: ryan
 *
 * Created on 27 April 2017, 00:01
 */

#ifndef STOCHASTIC_MOMENTUM_TRAINING_H
#define STOCHASTIC_MOMENTUM_TRAINING_H

#include "training/training_strategy.h"

#include <cassert>

class StochasticMomentumTraining: public TrainingStrategy{
public:
    StochasticMomentumTraining(){}
    void sample(ForwardMessageIterator first, ForwardMessageIterator last, DataWrapper* dataSet){
        // TODO complete or refactor into a separate strategy
    }
    
    void computeDeltaWeights(   shared_ptr<DNNGraphSettings> context,
                                float output, 
                                vector<float>& deltas, 
                                vector<float>& deltaWeights);
};


#endif /* STOCHASTIC_MOMENTUM_TRAINING_H */

