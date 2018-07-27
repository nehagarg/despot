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
	value(0),
        value_array(NULL){
}

ValuedAction::ValuedAction(int _action, double _value) :
	action(_action),
	value(_value),
        value_array(NULL){
}

ValuedAction::ValuedAction(int _action, std::vector<double>* _value_array):
action(_action),
	value_array(_value_array),
        value(0)
{
    
}
ValuedAction::~ValuedAction()
{
    /*if(value_array !=NULL)
    {
        std::cout << "Calling destructor \n";
        value_array->clear();
    }*/
}
ostream& operator<<(ostream& os, const ValuedAction& va) {
	os << "(" << va.action << ", " << va.value << ", [";
        if(va.value_array!=NULL)
        {
            for (int i = 0; i < va.value_array->size(); i++)
            {
                os << (*va.value_array)[i] << ", " ;
            }
        
        }
        os << "])" ;
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

	ValuedAction va = Value(particles, streams, history_, particles.size());

	for (int i = 0; i < particles.size(); i++)
		model_->Free(particles[i]);

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

ValuedAction POMCPScenarioLowerBound::Value(const vector<State*>& particles,
	RandomStreams& streams, History& history, int observation_particle_size) const {
	prior_->history(history);
	VNode* root = POMCP::CreateVNode(0, particles[0], prior_, model_);
	// Note that particles are assumed to be of equal weight
	for (int i = 0; i < particles.size(); i++) {
		State* particle = particles[i];
		State* copy = model_->Copy(particle);
		POMCP::Simulate(copy, streams, root, model_, prior_);
		model_->Free(copy);
	}

	ValuedAction va = POMCP::OptimalAction(root);
	va.value *= State::Weight(particles);
	delete root;
	return va;
}

/* =============================================================================
 * ParticleLowerBound class
 * =============================================================================*/

ParticleLowerBound::ParticleLowerBound(const DSPOMDP* model, Belief* belief) :
	ScenarioLowerBound(model, belief) {
}

ValuedAction ParticleLowerBound::Value(const vector<State*>& particles,
	RandomStreams& streams, History& history, int observation_particle_size) const {
	return Value(particles, observation_particle_size);
}

/* =============================================================================
 * TrivialParticleLowerBound class
 * =============================================================================*/

TrivialParticleLowerBound::TrivialParticleLowerBound(const DSPOMDP* model) :
	ParticleLowerBound(model) {
}

ValuedAction TrivialParticleLowerBound::Value(
	const vector<State*>& particles,int observation_particle_size) const {
	ValuedAction va = model_->GetMinRewardAction();
        
         
        if(Globals::config.track_alpha_vector)
        {
            va.value_array = new std::vector<double>(Globals::config.num_scenarios, va.value/ (1 - Globals::Discount()));
            //std::cout << "Lower bound returned" << va << std::endl;
            return va;
        }
       
	va.value *= State::Weight(particles)/ (1 - Globals::Discount()) ;
        
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
