#ifndef UPPER_BOUND_H
#define UPPER_BOUND_H

#include <vector>
#include <cassert>

#include <despot/random_streams.h>
#include <despot/core/history.h>

namespace despot {

class State;
class StateIndexer;
class DSPOMDP;
class Belief;
class MDP;
struct ValuedAction;

/* =============================================================================
 * ScenarioUpperBound class
 * =============================================================================*/

class ScenarioUpperBound {
public:
	ScenarioUpperBound();
	virtual ~ScenarioUpperBound();

	virtual void Init(const RandomStreams& streams);

	virtual double Value(const std::vector<State*>& particles,
		RandomStreams& streams, History& history, int observation_particle_size) const = 0;
        
        virtual void Value(const std::vector<State*>& particles,
		RandomStreams& streams, History& history, int observation_particle_size, 
                std::vector<double>& alpha_vector_upper_bound) const = 0;
};

/* =============================================================================
 * ParticleUpperBound class
 * =============================================================================*/

class ParticleUpperBound : public ScenarioUpperBound {
public:
	ParticleUpperBound();
	virtual ~ParticleUpperBound();

	/**
	 * Returns an upper bound to the maximum total discounted reward over an
	 * infinite horizon for the (unweighted) particle.
	 */
	virtual double Value(const State& state) const = 0;

	virtual double Value(const std::vector<State*>& particles,
		RandomStreams& streams, History& history, int observation_particle_size) const;

        virtual void Value(const std::vector<State*>& particles, RandomStreams& streams, History& history, int observation_particle_size, std::vector<double>& alpha_vector_upper_bound) const;

};

/* =============================================================================
 * TrivialParticleUpperBound class
 * =============================================================================*/

class TrivialParticleUpperBound: public ParticleUpperBound {
protected:
	const DSPOMDP* model_;
public:
	TrivialParticleUpperBound(const DSPOMDP* model);
	virtual ~TrivialParticleUpperBound();

	double Value(const State& state) const;

	virtual double Value(const std::vector<State*>& particles,
		RandomStreams& streams, History& history, int observation_particle_size) const;

        virtual void Value(const std::vector<State*>& particles, RandomStreams& streams, History& history, int observation_particle_size, std::vector<double>& alpha_vector_upper_bound) const;

};

/* =============================================================================
 * LookaheadUpperBound class
 * =============================================================================*/

class LookaheadUpperBound: public ScenarioUpperBound {
protected:
	const DSPOMDP* model_;
	const StateIndexer& indexer_;
	std::vector<std::vector<std::vector<double> > > bounds_;
	ParticleUpperBound* particle_upper_bound_;

public:
	LookaheadUpperBound(const DSPOMDP* model, const StateIndexer& indexer,
		ParticleUpperBound* bound);

	virtual void Init(const RandomStreams& streams);

	double Value(const std::vector<State*>& particles,
		RandomStreams& streams, History& history, int observation_particle_size) const;

        virtual void Value(const std::vector<State*>& particles, RandomStreams& streams, History& history, int observation_particle_size, std::vector<double>& alpha_vector_upper_bound) const;

};

/* =============================================================================
 * BeliefUpperBound class
 * =============================================================================*/

class BeliefUpperBound {
public:
	BeliefUpperBound();
	virtual ~BeliefUpperBound();

	virtual double Value(const Belief* belief, int observation_particle_size= -1) const = 0;
};

/* =============================================================================
 * TrivialBeliefUpperBound class
 * =============================================================================*/

class TrivialBeliefUpperBound: public BeliefUpperBound {
protected:
	const DSPOMDP* model_;
public:
	TrivialBeliefUpperBound(const DSPOMDP* model);

	double Value(const Belief* belief, int observation_particle_size=-1) const;
};

/* =============================================================================
 * MDPUpperBound class
 * =============================================================================*/

class MDPUpperBound: public ParticleUpperBound, public BeliefUpperBound {
protected:
	const MDP* model_;
	const StateIndexer& indexer_;
	std::vector<ValuedAction> policy_;

public:
	MDPUpperBound(const MDP* model, const StateIndexer& indexer);

  // shut off "hides overloaded virtual function" warning
  using ParticleUpperBound::Value;
	double Value(const State& state) const;

	double Value(const Belief* belief, int observation_particle_size=-1) const;
};

} // namespace despot

#endif
