#ifndef NODE_H
#define NODE_H

#include <despot/core/pomdp.h>
#include <despot/util/util.h>
#include <despot/random_streams.h>
#include <despot/util/logging.h>
#include "Python.h"

namespace despot {

class QNode;

/* =============================================================================
 * VNode class
 * =============================================================================*/

/**
 * A belief/value/AND node in the search tree.
 */
class VNode {
protected:
  std::vector<State*> particles_; // Used in DESPOT
  
	Belief* belief_; // Used in AEMS
	int depth_;
	QNode* parent_;
        QNode* common_parent_;
	OBS_TYPE edge_;
        

	std::vector<QNode*> children_;

        //std::vector<QNode*> common_children_; //Used in despot with alpha function update
        
        //std::vector<QNode*>* common_children_pointer_; //Used in despot with alpha function update
	ValuedAction default_move_; // Value and action given by default policy
	double lower_bound_;
	double upper_bound_;

	// For POMCP
	int count_; // Number of visits on the node
	double value_; // Value of the node

public:
	VNode* vstar;
	double likelihood; // Used in AEMS
	double utility_upper_bound;
        int observation_particle_size; //Used in despot with belief tracking
        std::vector<double> particle_weights; //used in despot with alpha function update
        std::vector<double> obs_probs; //used in despot with alpha function update
        //std::vector<double> upper_bound_alpha_vector_; //used in despot with alpha function update to store default upper bound vector for root node
        std::vector<double> lower_bound_alpha_vector_; //used in despot with alpha function to store best sibling lower bound vector
        ValuedAction lower_bound_alpha_vector; //used in despot with alpha function update
        ValuedAction upper_bound_alpha_vector; //used in despot with alpha function update
        bool extra_node; //used in despot with alpha function update
        VNode* obs_probs_holder;  //Used in despot with alpha function update
        PyObject* rnn_state; //Used in DESPOTWITHDEFAULTLEARNEDPOLICY
        PyObject* rnn_output;//Used in DESPOTWITHDEFAULTLEARNEDPOLICY

	VNode(std::vector<State*>& particles, int depth = 0, QNode* parent = NULL,
		OBS_TYPE edge = -1);
        VNode(int depth , QNode* parent ,
		OBS_TYPE edge );
        VNode(int depth , QNode* parent , QNode* common_parent,
		OBS_TYPE edge );
	VNode(Belief* belief, int depth = 0, QNode* parent = NULL, OBS_TYPE edge =
		-1);
	VNode(int count, double value, int depth = 0, QNode* parent = NULL,
		OBS_TYPE edge = -1);
	~VNode();

	Belief* belief() const;
	const std::vector<State*>& particles() const;
	void depth(int d);
	int depth() const;
	void parent(QNode* parent);
	QNode* parent();
        QNode* common_parent();
	OBS_TYPE edge();

	double Weight() const;

	const std::vector<QNode*>& children() const;
	std::vector<QNode*>& children();
	const QNode* Child(int action) const;
	QNode* Child(int action);
	int Size() const;
	int PolicyTreeSize() const;
        
        //const std::vector<QNode*>& common_children() const;
	//std::vector<QNode*>& common_children();
	const QNode* CommonChild(int action) const;
	QNode* CommonChild(int action);
        void CreateCommonQNode(int action);
        
        //std::vector<QNode*>* common_children_pointer();

	void default_move(ValuedAction move);
	ValuedAction default_move() const;
	void lower_bound(double value);
	double lower_bound() const;
	void upper_bound(double value);
	double upper_bound() const;
        double calculate_upper_bound() const;
        double calculate_lower_bound() const;
        
	bool IsLeaf();

	void Add(double val);
	void count(int c);
	int count() const;
	void value(double v);
	double value() const;

	void PrintTree(int depth = -1, std::ostream& os = std::cout);
	void PrintPolicyTree(int depth = -1, std::ostream& os = std::cout);

	void Free(const DSPOMDP& model);
        void FreeCommonParent(QNode* p);
};

/* =============================================================================
 * QNode class
 * =============================================================================*/

/**
 * A Q-node/AND-node (child of a belief node) of the search tree.
 */
class QNode {
protected:
	VNode* parent_;
	int edge_;
	std::map<OBS_TYPE, VNode*> children_;
        
	double lower_bound_;
	double upper_bound_;

	// For POMCP
	int count_; // Number of visits on the node
	double value_; // Value of the node

public:
	double default_value;
	double utility_upper_bound;
	double step_reward;
        std::vector<double> step_reward_vector;
	double likelihood;
	VNode* vstar;
        std::vector<QNode*> common_children_; //used in despot with alpha function update
        std::vector<State*> particles_; //Used for alpha function update algorithm
        std::vector<double> upper_bound_alpha_vector; //used in despot with alpha function update
        std::vector<double> lower_bound_alpha_vector; //used in despot with alpha function update
        std::vector<double> default_upper_bound_alpha_vector;
        ValuedAction default_move;
        
	QNode(VNode* parent, int edge);
        QNode(std::vector<State*>& particles);
	QNode(int count, double value);
	~QNode();
        
	void parent(VNode* parent);
	VNode* parent();
	int edge();
	std::map<OBS_TYPE, VNode*>& children();
	VNode* Child(OBS_TYPE obs);
	int Size() const;
	int PolicyTreeSize() const;

	double Weight() const;

	void lower_bound(double value);
	double lower_bound() const;
	void upper_bound(double value);
	double upper_bound() const;
        double calculate_upper_bound() const;
        double calculate_lower_bound() const;
        
	void Add(double val);
	void count(int c);
	int count() const;
	void value(double v);
	double value() const;
};

} // namespace despot

#endif
