
#include "tools/dnn_graph.h"

template<> string DNNGraphBuilder<ParallelDataNeuralNode>::type(){
    return "ParallelDataNeuralNode";
}

template<> string DNNGraphBuilder<NeuralNode>::type(){
    return "NeuralNode";
}

template<> string DNNGraphBuilder<AsyncNeuralNode>::type(){
    return "AsyncNeuralNode";
}