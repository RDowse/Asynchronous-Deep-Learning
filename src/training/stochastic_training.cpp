
#include "training/stochastic_training.h"
#include "graphs/dnn_graph_settings.h"

void StochasticTraining::computeDeltaWeights(   shared_ptr<DNNGraphSettings> context,
                                                float output, 
                                                vector<float>& deltas, 
                                                vector<float>& deltaWeights) {
    assert(deltas.size() == deltaWeights.size());
    for(int i = 0; i < deltas.size(); ++i)
        deltaWeights[i] = context->lr*deltas[i]*output; 
}
