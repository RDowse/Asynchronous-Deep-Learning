
#include "nodes/neural_node.h"

MessagePool<ForwardPropagationMessage>* NeuralNode::forwardMessagePool 
        = MessagePool<ForwardPropagationMessage>::getInstance();
MessagePool<BackwardPropagationMessage>* NeuralNode::backwardMessagePool  
        = MessagePool<BackwardPropagationMessage>::getInstance();
