
/* 
 * File:   message_pool.h
 * Author: ryan
 *
 * Created on 10 May 2017, 19:09
 */

#ifndef MESSAGE_POOL_H
#define MESSAGE_POOL_H

#include <tbb/concurrent_queue.h>
#include <queue>
#include <memory>

using namespace std;

template<typename TMessage>
class MessagePool{
    //queue<TMessage*> pool;
    std::shared_ptr<tbb::concurrent_bounded_queue<TMessage*>> pool;
    static MessagePool* instance;
    MessagePool();
public:
    ~MessagePool();
    static MessagePool* getInstance();
    TMessage* getMessage();
    void returnMessage(TMessage* msg);
};

#endif /* MESSAGE_POOL_H */
