
#include "nodes/node.h"
#include "messages/message.h"

int Node::curr_id = 0;

void Node::addEdge(Edge* e){
    if(e->dst->getId() == m_id) incomingEdges[e->src->getId()] = e;
    else if(e->src->getId() == m_id) outgoingEdges[e->dst->getId()] = e;
    else{
        throw runtime_error("this edges does not belong to this node \n");
        cout << "this edges does not belong to this node \n";  
        assert(0);
    }
}

void Node::send(vector<Message*>& msgs, list<Edge*>& activeEdges){
    for(unsigned i = 0; i < msgs.size(); ++i){
        Edge* e = outgoingEdges[msgs[i]->dst];
        if(e != NULL){
            assert( 0 == e->msgStatus );
            e->msg = msgs[i]; // Copy message into channel
            e->msgStatus = 
                static_cast<Edge::MessageStatus>(1 + e->getDelay()); // How long until it is ready?
            
            // add edge to the list of active edges
            activeEdges.push_back(e);
        } else {
            cout << "Node " << m_id << " is not connect to node id " << msgs[i]->dst << "\n";
            assert( e == NULL );
        }
    }
}

void Node::send(vector<Message*>& msgs, tbb::concurrent_queue<Edge*>& activeEdges){
    for(unsigned i = 0; i < msgs.size(); ++i){
        Edge* e = outgoingEdges[msgs[i]->dst];
        if(e != NULL){
            assert( 0 == e->msgStatus );
            e->msg = msgs[i]; // Copy message into channel
            e->msgStatus = 
                static_cast<Edge::MessageStatus>(1 + e->getDelay()); // How long until it is ready?
            
            // add edge to the list of active edges
            activeEdges.push(e);
        } else {
            cout << "Node " << m_id << " is not connect to node id " << msgs[i]->dst << "\n";
            assert( e == NULL );
        }
    }
}