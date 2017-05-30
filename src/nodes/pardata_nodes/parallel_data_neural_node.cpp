
#include "nodes/pardata_nodes/parallel_data_neural_node.h"

MessagePool<ForwardPropagationMessage>* ParallelDataNeuralNode::forwardMessagePool 
        = MessagePool<ForwardPropagationMessage>::getInstance();
MessagePool<BackwardPropagationMessage>* ParallelDataNeuralNode::backwardMessagePool  
        = MessagePool<BackwardPropagationMessage>::getInstance();
