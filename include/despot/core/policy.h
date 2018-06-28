#ifndef POLICY_H
#define POLICY_H

#include <vector>

#include <despot/random_streams.h>
#include <despot/core/lower_bound.h>
#include <despot/util/random.h>
#include <despot/core/history.h>

#include <string.h>
#include <queue>
#include <vector>
#include <stdlib.h>
#include <despot/core/globals.h>
#include <despot/core/pomdp.h>
#include <despot/core/node.h>

namespace despot {

class State;
class StateIndexer;
class StatePolicy;
class DSPOMDP;
class MMAPInferencer;
//class ParticleNode;

/* =============================================================================
 * ValuedAction struct
 * =============================================================================*/
/* Moved to node.h
struct ValuedAction {
	int action;
	double value;

	ValuedAction();
	ValuedAction(int _action, double _value);

	friend std::ostream& operator<<(std::ostream& os, const ValuedAction& va);
};
*/
/* =============================================================================
 * Policy class
 * =============================================================================*/

class Policy: public ScenarioLowerBound {
private:
	mutable int initial_depth_;
	ParticleLowerBound* particle_lower_bound_;

	ValuedAction RecursiveValue(ParticleNode* particle_node, std::vector<double>& particle_weights,
		std::vector<int> & obs_particle_ids, RandomStreams& streams, History& history, 
                int observation_particle_size) const;

public:
	Policy(const DSPOMDP* model, ParticleLowerBound* particle_lower_bound,
		Belief* belief = NULL);
	virtual ~Policy();

	void Reset();
	virtual int Action(ParticleNode* particle_node, std::vector<double>& particle_weights, std::vector<int> & obs_particle_ids, RandomStreams& streams,
		History& history, int observation_particle_size) const = 0;

	ParticleLowerBound* particle_lower_bound() const;

	ValuedAction Value(ParticleNode* particle_node, std::vector<double>& particle_weights, std::vector<int> & obs_particle_ids, RandomStreams& streams,
		History& history, int observation_particle_size) const;

	virtual ValuedAction Search();
};

/* =============================================================================
 * BlindPolicy class
 * =============================================================================*/

class BlindPolicy: public Policy {
private:
	int action_;

public:
	BlindPolicy(const DSPOMDP* model, int action, ParticleLowerBound*
		particle_lower_bound, Belief* belief = NULL);

	int Action(ParticleNode* particle_node, std::vector<double>& particle_weights, std::vector<int> & obs_particle_ids, RandomStreams& streams,
		History& history, int observation_particle_size) const;

	ValuedAction Search();
	void Update(int action, OBS_TYPE obs);
};

/* =============================================================================
 * RandomPolicy class
 * =============================================================================*/

class RandomPolicy: public Policy {
private:
	std::vector<double> action_probs_;

public:
	RandomPolicy(const DSPOMDP* model, ParticleLowerBound* ParticleLowerBound,
		Belief* belief = NULL);
	RandomPolicy(const DSPOMDP* model, const std::vector<double>& action_probs,
		ParticleLowerBound* ParticleLowerBound,
		Belief* belief = NULL);

	int Action(ParticleNode* particle_node, std::vector<double>& particle_weights, std::vector<int> & obs_particle_ids, RandomStreams& streams,
		History& history, int observation_particle_size) const;

	ValuedAction Search();
	void Update(int action, OBS_TYPE obs);
};

/* =============================================================================
 * ModeStatePolicy class
 * =============================================================================*/

class ModeStatePolicy: public Policy {
private:
	const StateIndexer& indexer_;
	const StatePolicy& policy_;
	mutable std::vector<double> state_probs_;

public:
	ModeStatePolicy(const DSPOMDP* model, const StateIndexer& indexer,
		const StatePolicy& policy, ParticleLowerBound* particle_lower_bound,
		Belief* belief = NULL);

	int Action(ParticleNode* particle_node, std::vector<double>& particle_weights, std::vector<int> & obs_particle_ids, RandomStreams& streams,
		History& history, int observation_particle_size) const;
};

/* =============================================================================
 * MMAPStatePolicy class
 * =============================================================================*/

class MMAPStatePolicy: public Policy { // Marginal MAP state policy
private:
	const MMAPInferencer& inferencer_;
	const StatePolicy& policy_;

public:
	MMAPStatePolicy(const DSPOMDP* model, const MMAPInferencer& inferencer,
		const StatePolicy& policy, ParticleLowerBound* particle_lower_bound,
		Belief* belief = NULL);

	int Action(ParticleNode* particle_node, std::vector<double>& particle_weights, std::vector<int> & obs_particle_ids, RandomStreams& streams,
		History& history, int observation_particle_size) const;
};

/* =============================================================================
 * MajorityActionPolicy class
 * =============================================================================*/

class MajorityActionPolicy: public Policy {
private:
	const StatePolicy& policy_;

public:
	MajorityActionPolicy(const DSPOMDP* model, const StatePolicy& policy,
		ParticleLowerBound* particle_lower_bound, Belief* belief = NULL);

	int Action(ParticleNode* particle_node, std::vector<double>& particle_weights, std::vector<int> & obs_particle_ids, RandomStreams& streams,
		History& history, int observation_particle_size) const;
};

} // namespace despot

#endif
