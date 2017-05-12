
#include "misc/message_pool.h"
#include "messages/backward_propagation_message.h"
#include "messages/forward_propagation_message.h"

#include <iostream>

template<typename TMessage>
MessagePool<TMessage>* MessagePool<TMessage>::instance = NULL;

template<typename TMessage>
MessagePool<TMessage>::MessagePool():pool(new tbb::concurrent_bounded_queue<TMessage*>()){}

template<typename TMessage>
MessagePool<TMessage>::~MessagePool(){
//    while(!pool.empty()){
//        delete pool.front();
//        pool.pop();
//    }
    while(!pool->empty()){
        TMessage* msg;
        pool->try_pop(msg);
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
//    if(pool.empty()) {
//        return new TMessage();
//    } else {
//        TMessage* msg = pool.front();
//        pool.pop();
//        return msg;
//    }
    if(pool->empty()) {
        return new TMessage();
    } else {
        TMessage* msg;
        pool->try_pop(msg);
        return msg;
    }
}

template<typename TMessage>
void MessagePool<TMessage>::returnMessage(TMessage* msg){
    // TODO: reset msg
    //pool.push(msg);
    pool->push(msg);
}

template class MessagePool<ForwardPropagationMessage>;
template class MessagePool<BackwardPropagationMessage>;