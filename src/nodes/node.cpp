
#include "nodes/node.h"

int Node::curr_id = 0;

void Node::send(vector<shared_ptr<Message>>& msgs, vector<Edge*>& edges){
    for(unsigned i=0; i < edges.size(); i++){
        assert( 0 == edges[i]->msgStatus );
        cout << edges[i]->dst->m_id << endl;
        edges[i]->msg = msgs[i]; // Copy message into channel
        edges[i]->msgStatus = 
            static_cast<Edge::MessageStatus>(1 + edges[i]->getDelay()); // How long until it is ready?
    }
}