#include <despot/core/lower_bound.h>
#include <despot/core/pomdp.h>
#include <despot/core/node.h>
#include <despot/solver/pomcp.h>

using namespace std;

namespace despot {

/* =============================================================================
 * ValuedAction class
 * =============================================================================*/

ValuedAction::ValuedAction() :
	action(-1),
	value(0) {
}

ValuedAction::ValuedAction(int _action, double _value) :
	action(_action),
	value(_value) {
}

ostream& operator<<(ostream& os, const ValuedAction& va) {
	os << "(" << va.action << ", " << va.value << ")";
	return os;
}

/* =============================================================================
 * ScenarioLowerBound class
 * =============================================================================*/

ScenarioLowerBound::ScenarioLowerBound(const DSPOMDP* model, Belief* belief) :
	Solver(model, belief) {
}

void ScenarioLowerBound::Init(const RandomStreams& streams) {
}

void ScenarioLowerBound::Reset() {
}

ValuedAction ScenarioLowerBound::Search() {
	RandomStreams streams(Globals::config.num_scenarios,
		Globals::config.search_depth);
	vector<State*> particles = belief_->Sample(Globals::config.num_scenarios);
        VNode *vnode = new VNode(particles);
	ValuedAction va = Value(vnode->particle_node_,vnode->particle_weight_,vnode->obs_particle_id_, streams, history_, vnode->observation_particle_size);

	for (int i = 0; i < particles.size(); i++)
            vnode->Free(*model_);
            //model_->Free(particles[i]);

	return va;
}

void ScenarioLowerBound::Learn(VNode* tree) {
}



/* =============================================================================
 * POMCPScenarioLowerBound class
 * =============================================================================*/

POMCPScenarioLowerBound::POMCPScenarioLowerBound(const DSPOMDP* model,
	POMCPPrior* prior,
	Belief* belief) :
	ScenarioLowerBound(model, belief),
	prior_(prior) {
	explore_constant_ = model_->GetMaxReward()
		- model_->GetMinRewardAction().value;
}

ValuedAction POMCPScenarioLowerBound::Value(ParticleNode* particle_node, std::vector<double>& particle_weights,
		std::vector<int> & obs_particle_ids, 
	RandomStreams& streams, History& history, int observation_particle_size) const {
	prior_->history(history);
	VNode* root = POMCP::CreateVNode(0, particle_node->particle(obs_particle_ids[0]), prior_, model_);
	// Note that particles are assumed to be of equal weight
	for (int i = 0; i < obs_particle_ids.size(); i++) {
		const State* particle = particle_node->particle(obs_particle_ids[i]);
		State* copy = model_->Copy(particle);
		POMCP::Simulate(copy, streams, root, model_, prior_);
		model_->Free(copy);
	}

	ValuedAction va = POMCP::OptimalAction(root);
	va.value *= State::Weight(particle_node, particle_weights, obs_particle_ids, observation_particle_size);
	delete root;
	return va;
}

/* =============================================================================
 * ParticleLowerBound class
 * =============================================================================*/

ParticleLowerBound::ParticleLowerBound(const DSPOMDP* model, Belief* belief) :
	ScenarioLowerBound(model, belief) {
}

ValuedAction ParticleLowerBound::Value(ParticleNode* particle_node, std::vector<double>& particle_weights,
		std::vector<int> & obs_particle_ids, 
	RandomStreams& streams, History& history, int observation_particle_size) const {
	return Value(particle_node, particle_weights,
		obs_particle_ids, observation_particle_size);
}

/* =============================================================================
 * TrivialParticleLowerBound class
 * =============================================================================*/

TrivialParticleLowerBound::TrivialParticleLowerBound(const DSPOMDP* model) :
	ParticleLowerBound(model) {
}

ValuedAction TrivialParticleLowerBound::Value(
ParticleNode* particle_node, std::vector<double>& particle_weights,
		std::vector<int> & obs_particle_ids, 	
int observation_particle_size) const {
	ValuedAction va = model_->GetMinRewardAction();
	va.value *= State::Weight(particle_node, particle_weights, obs_particle_ids, observation_particle_size) / (1 - Globals::Discount());
        if(observation_particle_size > 0)
        {
            va.value *= observation_particle_size*1.0/Globals::config.num_scenarios;
        }
	return va;
}

/* =============================================================================
 * BeliefLowerBound class
 * =============================================================================*/

BeliefLowerBound::BeliefLowerBound(const DSPOMDP* model, Belief* belief) :
	Solver(model, belief) {
}

ValuedAction BeliefLowerBound::Search() {
	return Value(belief_, -1);
}

void BeliefLowerBound::Learn(VNode* tree) {
}

/* =============================================================================
 * TrivialBeliefLowerBound class
 * =============================================================================*/

TrivialBeliefLowerBound::TrivialBeliefLowerBound(const DSPOMDP* model,
	Belief* belief) :
	BeliefLowerBound(model, belief) {
}

ValuedAction TrivialBeliefLowerBound::Value(const Belief* belief, int observation_particle_size) const {
	ValuedAction va = model_->GetMinRewardAction();
	va.value *= 1.0 / (1 - Globals::Discount());
	return va;
}

} // namespace despot
