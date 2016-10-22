# Asynchronous-Deep-Learning

## Specification
Deep learning has become a popular technique for machine learning,
particularly in the form of deep neural networks with many layers.
While deep learning can provide excellent classification performance,
a big draw-back is the amount of time needed to train them. Clusters
of GPUs provide one way of accelerating the learning process, but have
high capital and running costs.

This project will investigate the opportunities for learning asynchronously,
in order to target an event driven machine machines. In this mode, 
computation
is done by thousands of small cores exchanging tiny low-latency packets.
By avoiding the bulk synchronisation needed in a GPU, the idea is to both
increase throughput and reduce power. The downside is that algorithms need
to be re-written to tolerate weak ordering, with messages arriving
occasionally arriving out of sequence.

The goals of the project are:
- Look at techniques for mapping deep learning to an event driven
   asynchronous approach.
- Develop a concrete algorithm implementation of event driven
   programming using an existing event driven library.
- Evaluate the performance and scalability of the new algorithm
   in different architectures.

This is a research-oriented project, so while the overall goal is
well defined, the exact outcomes are not. No specific language
skills are needed beyond basic C++. Useful knowledge/skills would be:
- Machine learning/Neural Networks
- Optimisation
- Concurrent/parallel programming

