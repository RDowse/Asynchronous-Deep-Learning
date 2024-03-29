
#include "misc/message_pool.h"
#include "messages/backward_propagation_message.h"
#include "messages/forward_propagation_message.h"

#include <iostream>

template<typename TMessage>
MessagePool<TMessage>* MessagePool<TMessage>::instance = NULL;

template<typename TMessage>
MessagePool<TMessage>::MessagePool(){}

template<typename TMessage>
MessagePool<TMessage>::~MessagePool(){
    while(!pool.empty()){
        TMessage* msg;
        pool.try_pop(msg);
        delete msg;
    }
}

template<typename TMessage>
MessagePool<TMessage>* MessagePool<TMessage>::getInstance(){
    if(!instance){
        instance = new MessagePool<TMessage>();
    }
    return instance;
}

template<typename TMessage>
TMessage* MessagePool<TMessage>::getMessage(){
    TMessage* msg;
    if(!pool.try_pop(msg))
        msg = new TMessage();
    return msg;
}

template<typename TMessage>
void MessagePool<TMessage>::returnMessage(TMessage* msg){
    pool.push(msg);
}

template class MessagePool<ForwardPropagationMessage>;
template class MessagePool<BackwardPropagationMessage>;