#ifndef NODE_H
#define NODE_H

#include <despot/core/pomdp.h>
#include <despot/util/util.h>
#include <despot/random_streams.h>
#include <despot/util/logging.h>
#include <unordered_map>
#include "Python.h"

namespace despot {

class QNode;
class ParticleNode;
//struct ValuedAction;

/* =============================================================================
 * ValuedAction struct
 * =============================================================================*/

struct ValuedAction {
	int action;
	double value;

	ValuedAction();
	ValuedAction(int _action, double _value);

	friend std::ostream& operator<<(std::ostream& os, const ValuedAction& va);
};
  
/* =============================================================================
 * VNode class
 * =============================================================================*/

/**
 * A belief/value/AND node in the search tree.
 */
class VNode {
protected:
  //std::vector<State*> particles_; // Used in DESPOT //Replaced by particle_node
    
	Belief* belief_; // Used in AEMS
	int depth_;
	QNode* parent_;
	OBS_TYPE edge_;

	std::vector<QNode*> children_;

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
        ParticleNode* particle_node_; //Used in despot with belief tracking
        std::vector<double> particle_weight_;
        std::vector<int> obs_particle_id_;
        PyObject* rnn_state; //Used in DESPOTWITHDEFAULTLEARNEDPOLICY
        PyObject* rnn_output;//Used in DESPOTWITHDEFAULTLEARNEDPOLICY

        VNode(ParticleNode* particle_node, std::vector<int>& obs_particle_id, std::vector<double>& particle_weight, int obs_particle_size, int depth = 0, QNode* parent = NULL,
		OBS_TYPE edge = -1);
	VNode(std::vector<State*>& particles, int depth = 0, QNode* parent = NULL,
		OBS_TYPE edge = -1);
	VNode(Belief* belief, int depth = 0, QNode* parent = NULL, OBS_TYPE edge =
		-1);
	VNode(int count, double value, int depth = 0, QNode* parent = NULL,
		OBS_TYPE edge = -1);
	~VNode();

	Belief* belief() const;
	//const std::map<int,State*>& particles() const;
        int particles_size() const;
        const std::vector<int>& particle_ids() const;
	void depth(int d);
	int depth() const;
	void parent(QNode* parent);
	QNode* parent();
	OBS_TYPE edge();

	double Weight() const;

	const std::vector<QNode*>& children() const;
	std::vector<QNode*>& children();
	const QNode* Child(int action) const;
	QNode* Child(int action);
	int Size() const;
	int PolicyTreeSize() const;

	void default_move(ValuedAction move);
	ValuedAction default_move() const;
	void lower_bound(double value);
	double lower_bound() const;
	void upper_bound(double value);
	double upper_bound() const;

	bool IsLeaf();

	void Add(double val);
	void count(int c);
	int count() const;
	void value(double v);
	double value() const;

	void PrintTree(int depth = -1, std::ostream& os = std::cout);
	void PrintPolicyTree(int depth = -1, std::ostream& os = std::cout);

	void Free(const DSPOMDP& model);
        friend std::ostream& operator<<(std::ostream& os, const VNode& vnode);
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
	double likelihood;
	VNode* vstar;

	QNode(VNode* parent, int edge);
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

	void Add(double val);
	void count(int c);
	int count() const;
	void value(double v);
	double value() const;
};

class DoubleWithMinus1 {
private:
    double val;
public:
    DoubleWithMinus1() { val = -1;}
    void SetVal(double value)
    {
        val = value;
    }
    double GetVal()
    {
        return val;
    }
};
class ParticleNode {
    //friend class VNode;
    //friend class Policy;
    //friend class State;
    //friend std::ostream& operator<<(std::ostream& os, const VNode& vnode);
protected:
    //not using map because lookup in map takes more time than lookup in vector
    std::vector< OBS_TYPE> obs_;
    std::vector<double> reward_;
    std::vector<bool> terminal_;
    std::vector< std::unordered_map < OBS_TYPE, DoubleWithMinus1> > particle_obs_prob_;
    ParticleNode* parent_;
    std::vector<ParticleNode*> children_;
    std::vector<State*> particles_; 
    int edge_;
    int depth_;
    int num_particles;
public:
    
    ~ParticleNode() ;

    ParticleNode* Child(int action);
    void Child(int action, ParticleNode* p);
    ParticleNode* parent();
    void parent(ParticleNode* parent);
    void depth(int d);
    int depth() const;
    //const std::map<int, State*>& particles() const;
    State* particle(int i) const;
    const OBS_TYPE obs(int i) const;
    const double reward(int i) const;
    const bool terminal(int i) const;
    double obs_prob(int i, OBS_TYPE o);
    int particles_size() const;
    ParticleNode(std::vector<State*>& particles, int depth = 0, ParticleNode* parent = NULL,
		 int edge = -1);
    //ParticleNode(int depth = 0, ParticleNode* parent = NULL,
    //		 int edge = -1);
    void Add(State* particle, OBS_TYPE obs, double reward, bool terminal);
    void AddObsProb(int i, OBS_TYPE o, double prob);
    void Free(const DSPOMDP& model);
    static void particles_vector(const ParticleNode* particle_node, const std::vector<int>& obs_particle_ids, const int observation_particle_size, std::vector<State*>&  particle_vector, bool get_all = true);
};

} // namespace despot

#endif
