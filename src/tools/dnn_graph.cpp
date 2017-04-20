
#include "tools/dnn_graph.h"

map<string,string> DNNGraph::nodeColors = {
    std::pair<string,string>("Input","green2"),
    std::pair<string,string>("Hidden","yellow2"),
    std::pair<string,string>("Output","red2"),
    std::pair<string,string>("Sync","grey2"),
    std::pair<string,string>("Bias","blue2")
};