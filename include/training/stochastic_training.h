
/* 
 * File:   stochastic_training.h
 * Author: ryan
 *
 * Created on 25 April 2017, 15:15
 */

#ifndef STOCHASTIC_TRAINING_H
#define STOCHASTIC_TRAINING_H

#include "training/training_strategy.h"
#include "graphs/dnn_graph_settings.h"

#include <cassert>

class StochasticTraining: public TrainingStrategy{
public:
    StochasticTraining(){}
    void sample(ForwardMessageIterator first, ForwardMessageIterator last, DataWrapper* dataSet){
        // TODO complete or refactor into a separate strategy
    }
    
    void computeDeltaWeights(   shared_ptr<DNNGraphSettings> context,
                                float output, 
                                vector<float>& deltas, 
                                vector<float>& deltaWeights);
};

#endif /* STOCHASTIC_TRAINING_H */

