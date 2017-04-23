/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   dnn_graph.h
 * Author: ryan
 *
 * Created on 23 March 2017, 23:26
 */

#ifndef DNN_GRAPH_H
#define DNN_GRAPH_H

#include "graphs/dnn_graph_settings.h"

#include "nodes/node.h"
#include "nodes/hidden_node.h"
#include "nodes/input_node.h"
#include "nodes/output_node.h"
#include "nodes/bias_node.h"
#include "nodes/sync_node.h"

#include "misc/edge.h"

#include <algorithm>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <stdio.h>

using namespace std;

class DNNGraph{
    vector<Node*> nodes;
    vector<Edge*> edges;
    int nHLayers=0, nHidden=0, nInput=0, nOutput=0;
    int clusterCount = 0;
    float thickness = 0.2;
    
    // Map assigning colors to nodes for Graphviz
    static map<string,string> nodeColors;
public:    
    DNNGraph(int nHLayers, int nHidden, int nInput, int nOutput)
        :nHLayers(nHLayers),nHidden(nHidden),nInput(nInput),nOutput(nOutput)
    {
        auto settings = make_shared<DNNGraphSettings>();
        
        vector<Node*> prev_layer;
        vector<Node*> curr_layer;
        
        // Synchronisation node
        nodes.push_back(new SyncNode(settings));
        
        // Input nodes
        for(int i = 0; i < nInput; ++i){         
            prev_layer.push_back(new InputNode(settings));
            edges.push_back(
                new Edge(nodes.front(),prev_layer[i],1)
            );
            edges.push_back(
                new Edge(prev_layer[i],nodes.front(),1)
            );
        }
        
        // Hidden nodes
        for(int i = 0; i < nHLayers; ++i){
            // Bias node
            prev_layer.push_back(new BiasNode(settings));
            edges.push_back(new Edge(prev_layer.back(),nodes[0],1));
            edges.push_back(new Edge(nodes[0],prev_layer.back(),1));
            
            // Layer
            for(int j = 0; j < nHidden; ++j){
                curr_layer.push_back(new HiddenNode(settings));    
            }
            
            // Connect Edges
            for(int j = 0; j < prev_layer.size(); ++j){
                for(int k = 0; k < curr_layer.size(); ++k){
                    // fwd edge
                    edges.push_back(
                        new Edge(prev_layer[j],curr_layer[k],1)
                    );
                    // bck edge
                    edges.push_back(
                        new Edge(curr_layer[k],prev_layer[j],1)
                    );
                }
            }
            
            // Copy node layers
            nodes.insert(nodes.end(),prev_layer.begin(),prev_layer.end());
            swap(curr_layer,prev_layer);
            curr_layer.clear();
        }
        // Bias node
        prev_layer.push_back(new BiasNode(settings));
        edges.push_back(new Edge(prev_layer.back(),nodes[0],1));
        edges.push_back(new Edge(nodes[0],prev_layer.back(),1));
        
        // Output nodes
        for(int i = 0; i < nOutput; ++i){
            curr_layer.push_back(new OutputNode(settings));
        }
        // Connect Edges
        for(int j = 0; j < prev_layer.size(); ++j){
            for(int k = 0; k < curr_layer.size(); ++k){
                edges.push_back(
                    new Edge(prev_layer[j],curr_layer[k],1)
                );
                edges.push_back(
                    new Edge(curr_layer[k],prev_layer[j],1)
                );
            }
        }
        
        // Copy node layers
        nodes.insert(nodes.end(),prev_layer.begin(),prev_layer.end());
        nodes.insert(nodes.end(),curr_layer.begin(),curr_layer.end());
        
        // Connect output to sync node
        for(int i = 0; i < nOutput; ++i){
            edges.push_back(
                new Edge(nodes.front(),curr_layer[i],1)
            );
            edges.push_back(
                new Edge(curr_layer[i],nodes.front(),1)
            );
        }
    }
    
    ~DNNGraph(){
        for(auto it = nodes.begin(); it != nodes.end(); it++)
            delete (*it);
        nodes.clear();
        for(auto it = edges.begin(); it != edges.end(); it++)
            delete (*it);
        edges.clear();
    }
    
    void writeGraph(const string& path){
        ofstream file;
        file.open(path);
        if(file.is_open()){
            file<<"POETSGraph\n";
            file<<"DNN"<<"\n";      // TYPE
            
            file<<"BeginHeader\n";
            file<<"0"<<"\n";        // PARAMETER COUNT
            file<<"NULL"<<"\n";     // PARAMETERS
            file<<nodes.size()<<" "<<edges.size()<<"\n";
            file<<"EndHeader\n";   
            
            file<<"BeginNodes\n";
            for(auto n: nodes){
                file<<n->getType()<<" "<<n->getId()<<"\n";
            }
            file<<"EndNodes\n";
            
            file<<"BeginEdges\n";
            for(auto e: edges){
            file<<e->src->getId()<<" "<<e->dst->getId()<<" "<<e->getDelay()<<"\n";
            }
            file<<"EndEdges\n";
        } else {
            std::cout<< "Error opening file: %s" << path;
        }
        file.close();
    }
 
    void writeGraphviz(const string& path){
        FILE* file;
        file = fopen(path.c_str(),"w");
        if(file){
            int index = 0;
            fprintf(file,"digraph G {\n");
            fprintf(file,"rankdir=LR\n");
            fprintf(file,"splines=line\n");
            fprintf(file,"node [fixedsize=true, label=\"\"];\n");
            
            // Input sync node
            printGraphvizCluster(file,0,
                    1,nodes[0]->getType());
            index++;
            
            // Input nodes
            index += nInput+1;
            printGraphvizCluster(file,1,
                    index,nodes[1]->getType());
            
            // Hidden nodes
            for(int i = 0; i < nHLayers; ++i){
                printGraphvizCluster(file,index+i*(nHidden+1),
                        index+(nHidden+1),nodes[index]->getType());
                index += nHidden+1;
            }
            
            // Output nodes
            printGraphvizCluster(file,index,
                    index+nOutput,nodes[index]->getType());
            index+=nOutput;
            
            fprintf(file,"\n");
            printGraphvizConnections(file);
            fprintf(file,"}");
        } else {
            std::cout<< "Error opening file: %s" << path;
        }
        fclose(file);
    }
 
private:
    void printGraphvizCluster(FILE* file, int start, int end, string prefix){
        fprintf(file,"subgraph cluster_%d{\n",clusterCount++);
        fprintf(file,"    color=%s\n","white");
        fprintf(file,"    node [style=solid,color=%s,shape=circle];\n",
                nodeColors[nodes[start]->getType()].c_str());
        //fprintf(file,"    edge [style=\"invisible\",dir=\"none\"];\n");
        //fprintf(file,"    rank=\"same\";\n     ");
        
        for(int i = start; i < end; ++i) // TODO refactor for node specific labeling
            if(nodes[i]->getType()=="Bias") fprintf(file,"%s%d [ label=\"+1\" ];",nodes[i]->getType().c_str(),nodes[i]->getId());
        
        for(int i = start; i < end-1; ++i)
            fprintf(file,"%s%d ",nodes[i]->getType().c_str(),nodes[i]->getId());
        fprintf(file,"%s%d;\n",nodes[end-1]->getType().c_str(),nodes[end-1]->getId()); 
        fprintf(file,"    label = \"%s layer\";\n}\n",prefix.c_str());
    }
    
    void printGraphvizConnections(FILE* file){
        for(auto e: edges){
            if(e->src->getType() == "Sync" || e->dst->getType() == "Sync")
                fprintf(file,"%s%d -> %s%d [penwidth=%f constraint=\"false\"]\n ",e->src->getType().c_str(),e->src->getId(),
                                e->dst->getType().c_str(),e->dst->getId(),thickness);
            else
                fprintf(file,"%s%d -> %s%d [penwidth=%f]\n",e->src->getType().c_str(),e->src->getId(),
                                e->dst->getType().c_str(),e->dst->getId(),thickness);
        }
    }
};

#endif /* DNN_GRAPH_H */

