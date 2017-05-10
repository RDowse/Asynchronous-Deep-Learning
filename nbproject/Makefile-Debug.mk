#
# Generated Makefile - do not edit!
#
# Edit the Makefile in the project folder instead (../Makefile). Each target
# has a -pre and a -post target defined where you can add customized code.
#
# This makefile implements configuration specific macros and targets.


# Environment
MKDIR=mkdir
CP=cp
GREP=grep
NM=nm
CCADMIN=CCadmin
RANLIB=ranlib
CC=gcc
CCC=g++
CXX=g++
FC=gfortran
AS=as

# Macros
CND_PLATFORM=GNU-Linux
CND_DLIB_EXT=so
CND_CONF=Debug
CND_DISTDIR=dist
CND_BUILDDIR=build

# Include project Makefile
include Makefile

# Object Directory
OBJECTDIR=${CND_BUILDDIR}/${CND_CONF}/${CND_PLATFORM}

# Object Files
OBJECTFILES= \
	${OBJECTDIR}/src/main.o \
	${OBJECTDIR}/src/messages/backward_propagation_message.o \
	${OBJECTDIR}/src/messages/forward_propagation_message.o \
	${OBJECTDIR}/src/misc/node_factory.o \
	${OBJECTDIR}/src/nodes/bias_node.o \
	${OBJECTDIR}/src/nodes/block_nodes/block_hidden_node.o \
	${OBJECTDIR}/src/nodes/block_nodes/block_input_node.o \
	${OBJECTDIR}/src/nodes/block_nodes/block_output_node.o \
	${OBJECTDIR}/src/nodes/block_nodes/block_sync_node.o \
	${OBJECTDIR}/src/nodes/hidden_node.o \
	${OBJECTDIR}/src/nodes/input_node.o \
	${OBJECTDIR}/src/nodes/node.o \
	${OBJECTDIR}/src/nodes/output_node.o \
	${OBJECTDIR}/src/nodes/sync_node.o \
	${OBJECTDIR}/src/states/backward_train_state.o \
	${OBJECTDIR}/src/states/forward_train_state.o \
	${OBJECTDIR}/src/states/predict_state.o \
	${OBJECTDIR}/src/states/state.o \
	${OBJECTDIR}/src/tools/clock.o \
	${OBJECTDIR}/src/tools/dnn_graph.o \
	${OBJECTDIR}/src/tools/logging.o \
	${OBJECTDIR}/src/training/stochastic_momentum_training.o \
	${OBJECTDIR}/src/training/stochastic_training.o


# C Compiler Flags
CFLAGS=

# CC Compiler Flags
CCFLAGS=
CXXFLAGS=

# Fortran Compiler Flags
FFLAGS=

# Assembler Flags
ASFLAGS=

# Link Libraries and Options
LDLIBSOPTIONS=

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS}
	"${MAKE}"  -f nbproject/Makefile-${CND_CONF}.mk ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/asynchronous-deep-learning

${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/asynchronous-deep-learning: ${OBJECTFILES}
	${MKDIR} -p ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}
	${LINK.cc} -o ${CND_DISTDIR}/${CND_CONF}/${CND_PLATFORM}/asynchronous-deep-learning ${OBJECTFILES} ${LDLIBSOPTIONS} -ltbb

${OBJECTDIR}/src/main.o: src/main.cpp
	${MKDIR} -p ${OBJECTDIR}/src
	${RM} "$@.d"
	$(COMPILE.cc) -O3 -Iinclude -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/main.o src/main.cpp

${OBJECTDIR}/src/messages/backward_propagation_message.o: src/messages/backward_propagation_message.cpp
	${MKDIR} -p ${OBJECTDIR}/src/messages
	${RM} "$@.d"
	$(COMPILE.cc) -O3 -Iinclude -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/messages/backward_propagation_message.o src/messages/backward_propagation_message.cpp

${OBJECTDIR}/src/messages/forward_propagation_message.o: src/messages/forward_propagation_message.cpp
	${MKDIR} -p ${OBJECTDIR}/src/messages
	${RM} "$@.d"
	$(COMPILE.cc) -O3 -Iinclude -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/messages/forward_propagation_message.o src/messages/forward_propagation_message.cpp

${OBJECTDIR}/src/misc/node_factory.o: src/misc/node_factory.cpp
	${MKDIR} -p ${OBJECTDIR}/src/misc
	${RM} "$@.d"
	$(COMPILE.cc) -O3 -Iinclude -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/misc/node_factory.o src/misc/node_factory.cpp

${OBJECTDIR}/src/nodes/bias_node.o: src/nodes/bias_node.cpp
	${MKDIR} -p ${OBJECTDIR}/src/nodes
	${RM} "$@.d"
	$(COMPILE.cc) -O3 -Iinclude -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/nodes/bias_node.o src/nodes/bias_node.cpp

${OBJECTDIR}/src/nodes/block_nodes/block_hidden_node.o: src/nodes/block_nodes/block_hidden_node.cpp
	${MKDIR} -p ${OBJECTDIR}/src/nodes/block_nodes
	${RM} "$@.d"
	$(COMPILE.cc) -O3 -Iinclude -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/nodes/block_nodes/block_hidden_node.o src/nodes/block_nodes/block_hidden_node.cpp

${OBJECTDIR}/src/nodes/block_nodes/block_input_node.o: src/nodes/block_nodes/block_input_node.cpp
	${MKDIR} -p ${OBJECTDIR}/src/nodes/block_nodes
	${RM} "$@.d"
	$(COMPILE.cc) -O3 -Iinclude -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/nodes/block_nodes/block_input_node.o src/nodes/block_nodes/block_input_node.cpp

${OBJECTDIR}/src/nodes/block_nodes/block_output_node.o: src/nodes/block_nodes/block_output_node.cpp
	${MKDIR} -p ${OBJECTDIR}/src/nodes/block_nodes
	${RM} "$@.d"
	$(COMPILE.cc) -O3 -Iinclude -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/nodes/block_nodes/block_output_node.o src/nodes/block_nodes/block_output_node.cpp

${OBJECTDIR}/src/nodes/block_nodes/block_sync_node.o: src/nodes/block_nodes/block_sync_node.cpp
	${MKDIR} -p ${OBJECTDIR}/src/nodes/block_nodes
	${RM} "$@.d"
	$(COMPILE.cc) -O3 -Iinclude -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/nodes/block_nodes/block_sync_node.o src/nodes/block_nodes/block_sync_node.cpp

${OBJECTDIR}/src/nodes/hidden_node.o: src/nodes/hidden_node.cpp
	${MKDIR} -p ${OBJECTDIR}/src/nodes
	${RM} "$@.d"
	$(COMPILE.cc) -O3 -Iinclude -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/nodes/hidden_node.o src/nodes/hidden_node.cpp

${OBJECTDIR}/src/nodes/input_node.o: src/nodes/input_node.cpp
	${MKDIR} -p ${OBJECTDIR}/src/nodes
	${RM} "$@.d"
	$(COMPILE.cc) -O3 -Iinclude -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/nodes/input_node.o src/nodes/input_node.cpp

${OBJECTDIR}/src/nodes/node.o: src/nodes/node.cpp
	${MKDIR} -p ${OBJECTDIR}/src/nodes
	${RM} "$@.d"
	$(COMPILE.cc) -O3 -Iinclude -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/nodes/node.o src/nodes/node.cpp

${OBJECTDIR}/src/nodes/output_node.o: src/nodes/output_node.cpp
	${MKDIR} -p ${OBJECTDIR}/src/nodes
	${RM} "$@.d"
	$(COMPILE.cc) -O3 -Iinclude -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/nodes/output_node.o src/nodes/output_node.cpp

${OBJECTDIR}/src/nodes/sync_node.o: src/nodes/sync_node.cpp
	${MKDIR} -p ${OBJECTDIR}/src/nodes
	${RM} "$@.d"
	$(COMPILE.cc) -O3 -Iinclude -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/nodes/sync_node.o src/nodes/sync_node.cpp

${OBJECTDIR}/src/states/backward_train_state.o: src/states/backward_train_state.cpp
	${MKDIR} -p ${OBJECTDIR}/src/states
	${RM} "$@.d"
	$(COMPILE.cc) -O3 -Iinclude -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/states/backward_train_state.o src/states/backward_train_state.cpp

${OBJECTDIR}/src/states/forward_train_state.o: src/states/forward_train_state.cpp
	${MKDIR} -p ${OBJECTDIR}/src/states
	${RM} "$@.d"
	$(COMPILE.cc) -O3 -Iinclude -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/states/forward_train_state.o src/states/forward_train_state.cpp

${OBJECTDIR}/src/states/predict_state.o: src/states/predict_state.cpp
	${MKDIR} -p ${OBJECTDIR}/src/states
	${RM} "$@.d"
	$(COMPILE.cc) -O3 -Iinclude -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/states/predict_state.o src/states/predict_state.cpp

${OBJECTDIR}/src/states/state.o: src/states/state.cpp
	${MKDIR} -p ${OBJECTDIR}/src/states
	${RM} "$@.d"
	$(COMPILE.cc) -O3 -Iinclude -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/states/state.o src/states/state.cpp

${OBJECTDIR}/src/tools/clock.o: src/tools/clock.cpp
	${MKDIR} -p ${OBJECTDIR}/src/tools
	${RM} "$@.d"
	$(COMPILE.cc) -O3 -Iinclude -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/tools/clock.o src/tools/clock.cpp

${OBJECTDIR}/src/tools/dnn_graph.o: src/tools/dnn_graph.cpp
	${MKDIR} -p ${OBJECTDIR}/src/tools
	${RM} "$@.d"
	$(COMPILE.cc) -O3 -Iinclude -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/tools/dnn_graph.o src/tools/dnn_graph.cpp

${OBJECTDIR}/src/tools/logging.o: src/tools/logging.cpp
	${MKDIR} -p ${OBJECTDIR}/src/tools
	${RM} "$@.d"
	$(COMPILE.cc) -O3 -Iinclude -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/tools/logging.o src/tools/logging.cpp

${OBJECTDIR}/src/training/stochastic_momentum_training.o: src/training/stochastic_momentum_training.cpp
	${MKDIR} -p ${OBJECTDIR}/src/training
	${RM} "$@.d"
	$(COMPILE.cc) -O3 -Iinclude -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/training/stochastic_momentum_training.o src/training/stochastic_momentum_training.cpp

${OBJECTDIR}/src/training/stochastic_training.o: src/training/stochastic_training.cpp
	${MKDIR} -p ${OBJECTDIR}/src/training
	${RM} "$@.d"
	$(COMPILE.cc) -O3 -Iinclude -I/usr/include/eigen3 -std=c++11 -MMD -MP -MF "$@.d" -o ${OBJECTDIR}/src/training/stochastic_training.o src/training/stochastic_training.cpp

# Subprojects
.build-subprojects:

# Clean Targets
.clean-conf: ${CLEAN_SUBPROJECTS}
	${RM} -r ${CND_BUILDDIR}/${CND_CONF}

# Subprojects
.clean-subprojects:

# Enable dependency checking
.dep.inc: .depcheck-impl

include .dep.inc
