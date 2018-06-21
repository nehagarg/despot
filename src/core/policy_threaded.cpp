#include <despot/core/policy.h>
#include <despot/core/pomdp.h>
#include <unistd.h>
#include <thread>
#include <future>

using namespace std;

namespace despot {

/* =============================================================================
 * Policy class
 * =============================================================================*/

Policy::Policy(const DSPOMDP* model, ParticleLowerBound* particle_lower_bound,
		Belief* belief) :
	ScenarioLowerBound(model, belief),
	particle_lower_bound_(particle_lower_bound) {
	assert(particle_lower_bound_ != NULL);
}

Policy::~Policy() {
    }

    void PolicyThreadTest(int &a, int b) {
        a = 1;
        return;
    }


void PolicyThreadStep(State& state, double random_num, int action, double& reward, OBS_TYPE& obs, bool& terminal, const DSPOMDP* model) {
  //terminal =  true;
  //return;
            model->Step(state, random_num, action, reward, obs);
}

ValuedAction Policy::Value(const vector<State*>& particles,
	RandomStreams& streams, History& history, int observation_particle_size) const {
    //std::cout<< "Observation particle size" << observation_particle_size << std::endl;
	vector<State*> copy;
        //vector<double> copy_weight;
        /*if(observation_particle_size > 0)
        {
            for (int i = 0; i < particles.size(); i++)
            copy_weight.push_back(particles[i]->weight);
        }*/
        //std::cout << "Num active particles 0 :" << model_->NumActiveParticles() << std::endl;
	for (int i = 0; i < particles.size(); i++)
		copy.push_back(model_->Copy(particles[i]));
        
        //std::cout << "Num active particles 1 :" << model_->NumActiveParticles() << std::endl;
	initial_depth_ = history.Size();
	ValuedAction va = RecursiveValue(copy, streams, history, observation_particle_size );
        
        //std::cout << "Num active particles 2 :" << model_->NumActiveParticles() << std::endl;
        //Belief particles get freed in recursive value function
        int free_particle_size = copy.size();
        /*if(observation_particle_size > 0)
        {
            free_particle_size = observation_particle_size;
        }*/
        //std::cout<< "Free particle size" << free_particle_size << std::endl;
        if(observation_particle_size < 0)
        {
	for (int i = 0; i < free_particle_size; i++)
		model_->Free(copy[i]);
        }
        
        //std::cout << "Num active particles 3 :" << model_->NumActiveParticles() << std::endl;
        //std::cout<< "Valued action" << va << std::endl;

	return va;
}

ValuedAction Policy::RecursiveValue(const vector<State*>& particles,
	RandomStreams& streams, History& history, 
        int obs_particle_size) const {
    //std::cout<< "Recursive Observation particle size" << obs_particle_size << std::endl;
        /*if(obs_particle_size > 0)
            {
                for (int i = 0; i < particles.size(); i++)
                {
                    //particles[i]->weight = weight_vector[i];
                    std::cout << i << ":" << *particles[i] << std::endl;
                }
                
		
            }*/
	if (streams.Exhausted()
		|| (history.Size() - initial_depth_
			>= Globals::config.max_policy_sim_len)) {
            ValuedAction ans = particle_lower_bound_->Value(particles, obs_particle_size);
            if(obs_particle_size > 0)
            {
                for (int i = 0; i < particles.size(); i++)
                {
                    model_->Free(particles[i]);
                }
                    
            }
            return ans;
            
            
		
	} else {
		int action = Action(particles, streams, history);
                if(action < 0)
                {   
                   ValuedAction ans =  particle_lower_bound_->Value(particles, obs_particle_size);
                    if(obs_particle_size > 0)
                    {
                        for (int i = 0; i < particles.size(); i++)
                        {
                            model_->Free(particles[i]);
                        }

                    }
                    return ans;
                    //return particle_lower_bound_->Value(particles);
                }
                
		double value = 0;
                int observation_particle_size = obs_particle_size;
                
                
                
                std::map<OBS_TYPE, std::vector<State*> > partitions_belief_; //Stores belief particles
		map<OBS_TYPE, vector<State*> > partitions;
                //map<OBS_TYPE, vector<double> > partitions_weight;
                //map<OBS_TYPE, vector<double> > partitions_belief_weight;
                
                
                //std::vector<bool> terminals;
                //std::vector<OBS_TYPE> obss;
                //std::vector<double> rewards;
                bool terminals[particles.size()];
                OBS_TYPE obss[particles.size()];
                double rewards[particles.size()];
                std::vector<thread> threads;
                //terminals.resize(particles.size());
                //obss.resize(particles.size());
                //rewards.resize(particles.size());
		for (int i = 0; i < particles.size(); i++){
                    State* particle = particles[i];
                   
                    double random_num = streams.Entry(particle->scenario_id);
                   threads.push_back(std::thread(PolicyThreadStep, std::ref(*(particle))
				, random_num,
                           action, std::ref(rewards[i]), std::ref(obss[i]), std::ref(terminals[i]), model_));
                   //int a,b;
                    //std::thread t1(PolicyThreadTest, a, b);
                    //threads.push_back(std::move(t1));
                 // threads.push_back(std::thread(PolicyThreadTest, std::ref(a), b));
                }
                for (int i = 0; i < particles.size(); i++){
                    threads[i].join();
                }
                
                
                OBS_TYPE obs;
		double reward;
		for (int i = 0; i < particles.size(); i++) {
			State* particle = particles[i];
                        bool terminal = terminals[i];
                        reward = rewards[i];
                        obs = obss[i];
			//bool terminal = model_->Step(*particle,
			//	streams.Entry(particle->scenario_id), action, reward, obs);

			value += reward * particle->weight;

			if (!terminal) {
                            if(i< observation_particle_size && obs_particle_size>0)
                            {
                                if(partitions.count(obs) == 0)
                                {
                                    for (std::map<OBS_TYPE, std::vector<State*> >::iterator it = partitions.begin();
                        it != partitions.end(); it++)
                                    {
                                        OBS_TYPE obs_key = it->first;
                                        for (int j = 0; j< partitions[obs_key].size();j++)
                                        {
                                            State* copy1 = model_->Copy(partitions[obs_key][j]);
                                            partitions_belief_[obs].push_back(copy1);
                                            //partitions_belief_weight[obs].push_back(partitions[obs_key][j]->weight);
                                        }
                                    }
                                }
				
                            }
                            if(i < observation_particle_size || obs_particle_size<0)
                            {
                                partitions[obs].push_back(particle);
                                /*if(obs_particle_size > 0)
                                {
                                    partitions_weight[obs].push_back(particle->weight);
                                }*/
                            }
                            
                            
                            
                            if(obs_particle_size > 0)
                            {
                                int first_push = 0;
                                for (std::map<OBS_TYPE, std::vector<State*> >::iterator it = partitions.begin();
		it != partitions.end(); it++) {
                                    OBS_TYPE obs_key = it->first;
                                    if(i>=observation_particle_size || obs_key!=obs)
                                    {
                                        if(i>=observation_particle_size)
                                        {
                                            if(first_push == 0)
                                            {
                                                partitions_belief_[obs_key].push_back(particle);
                                                first_push++;
                                            }
                                            else
                                            {
                                                State* copy1 = model_->Copy(particle);
                                                partitions_belief_[obs_key].push_back(copy1);
                                            }
                                        }
                                        else
                                        {
                                            State* copy1 = model_->Copy(particle);
                                            partitions_belief_[obs_key].push_back(copy1);
                                        }
                                        //partitions_belief_weight[obs_key].push_back(particle->weight);
                                    }
                            
                                }
                                if(i>=observation_particle_size && first_push == 0)
                                {
                                    //Observation particles reached terminal state. So belief particles have to be deleted
                                    model_->Free(particle);
                                }
                            }
			}
                        else
                        {
                            if(obs_particle_size > 0)
                            {
                                model_->Free(particle);// std::cout << "terminal particle" << std::endl;
                        
                            }
                        }
                        /*if(i>= observation_particle_size && obs_particle_size>0)
                        {
                            model_->Free(particle);
                        }*/   
		}
                if(obs_particle_size > 0)
                {
                    value = value*observation_particle_size*1.0/Globals::config.num_scenarios;
                }
                /*std::cout << "Depth: " << history.Size() << " particles " << model_->NumActiveParticles() << std::endl;
                int total_obs_partcles = 0;
                int total_belief_particles = 0;
                for (map<OBS_TYPE, vector<State*> >::iterator it = partitions.begin();
			it != partitions.end(); it++) {
			OBS_TYPE obs = it->first;
                        //std::cout << "Observation " << obs << std::endl;
                        //std::cout << "Observation particles " << partitions[obs].size() << std::endl;
                        
                        //std::cout << "Belief particles " << partitions_belief_[obs].size() << std::endl;
                        total_obs_partcles += partitions[obs].size();
                        total_belief_particles += partitions_belief_[obs].size();
                }
                std::cout << "Observation particles " << total_obs_partcles << " Belief particles " << total_belief_particles << std::endl;
                
                 */
                /*
                for (map<OBS_TYPE, vector<State*> >::iterator it = partitions.begin();
			it != partitions.end(); it++) {
			OBS_TYPE obs = it->first;
                        std::cout << "Observation particles " << std::endl;
                        for(int i = 0; i < partitions[obs].size();i++)
                            {
                            std::cout << *partitions[obs][i] << std::endl;
                                model_->PrintObs(*partitions[obs][i], obs);
                               // std::cout << "Weight = " << partitions_weight[obs][i] << std::endl;
                        }
                        std::cout << "Belief particles " << std::endl;
                        for(int i = 0; i < partitions_belief_[obs].size();i++)
                            {
                            std::cout << *partitions_belief_[obs][i] << std::endl;
                                model_->PrintObs(*partitions_belief_[obs][i], obs);
                                //std::cout << "Weight = " << partitions_belief_weight[obs][i] << std::endl;
                        }
                        
                }*/
                
                //std::cout << "Depth: " << history.Size() << " Got reward value :(" << action << "," << value << ")\n";
		
                for (map<OBS_TYPE, vector<State*> >::iterator it = partitions.begin();
			it != partitions.end(); it++) {
			OBS_TYPE obs = it->first;
                        
                        int observation_particle_size_ = obs_particle_size;
                        if(obs_particle_size > 0)
                        {
                            observation_particle_size_ = partitions[obs].size();
                            partitions[obs].insert(partitions[obs].end(),partitions_belief_[obs].begin(),partitions_belief_[obs].end());
                            //partitions_weight[obs].insert(partitions_weight[obs].end(),
                            //        partitions_belief_weight[obs].begin(),
                            //        partitions_belief_weight[obs].end());
                            double total_weight = 0;
                            for(int i = 0; i < partitions[obs].size();i++)
                            {
                                //model_->PrintObs(*partitions[obs][i], obs);
                                double prob = model_->ObsProb(obs, *partitions[obs][i], action);
                                //std::cout << "Obs Prob:" <<  prob << " " << *partitions[obs][i] << std::endl;
                
                                 // Terminal state is not required to be explicitly represented and may not have any observation
                                //partitions_weight[obs][i] *= prob;
                                //total_weight += partitions_weight[obs][i];
                                partitions[obs][i]->weight *=prob;
                                total_weight += partitions[obs][i]->weight;
                        //Total weight should not be zero as one particle actually produced that observation
                            }
                            for(int i = 0; i < partitions[obs].size();i++)
                            {
                                //partitions_weight[obs][i]= partitions_weight[obs][i]/total_weight;
                                partitions[obs][i]->weight = partitions[obs][i]->weight/total_weight;
                            }
                
                        }
                        //std::cout<< "Recursive Partitioned Observation size" << observation_particle_size_<< std::endl;
			history.Add(action, obs);
			streams.Advance();
			ValuedAction va = RecursiveValue(it->second, streams, history, observation_particle_size_);
                        //std::cout << "Depth: " << history.Size() << " Got value :" << va << "\n";
			//value += Globals::Discount() * va.value*observation_particle_size_/observation_particle_size;
                        value += Globals::Discount() * va.value;
			streams.Back();
			history.RemoveLast();
		}
               // std::cout << "Depth: " << history.Size() << " particles " << model_->NumActiveParticles() << std::endl;
                
		return ValuedAction(action, value);
	}
}

void Policy::Reset() {
}

ParticleLowerBound* Policy::particle_lower_bound() const {
	return particle_lower_bound_;
}

ValuedAction Policy::Search() {
	RandomStreams streams(Globals::config.num_scenarios,
		Globals::config.search_depth);
	vector<State*> particles = belief_->Sample(Globals::config.num_scenarios);

	int action = Action(particles, streams, history_);
	double dummy_value = Globals::NEG_INFTY;

	for (int i = 0; i < particles.size(); i++)
		model_->Free(particles[i]);

	return ValuedAction(action, dummy_value);
}

/* =============================================================================
 * BlindPolicy class
 * =============================================================================*/

BlindPolicy::BlindPolicy(const DSPOMDP* model, int action, ParticleLowerBound* 
	bound, Belief* belief) :
	Policy(model, bound, belief),
	action_(action) {
}

int BlindPolicy::Action(const vector<State*>& particles, RandomStreams& streams,
	History& history) const {
	return action_;
}

ValuedAction BlindPolicy::Search() {
	double dummy_value = Globals::NEG_INFTY;
	return ValuedAction(action_, dummy_value);
}

void BlindPolicy::Update(int action, OBS_TYPE obs) {
}

/* =============================================================================
 * RandomPolicy class
 * =============================================================================*/

RandomPolicy::RandomPolicy(const DSPOMDP* model, ParticleLowerBound* bound,
	Belief* belief) :
	Policy(model, bound, belief) {
}

RandomPolicy::RandomPolicy(const DSPOMDP* model,
	const vector<double>& action_probs,
	ParticleLowerBound* bound, Belief* belief) :
	Policy(model, bound, belief),
	action_probs_(action_probs) {
	double sum = 0;
	for (int i = 0; i < action_probs.size(); i++)
		sum += action_probs[i];
	assert(fabs(sum - 1.0) < 1.0e-8);
}

int RandomPolicy::Action(const vector<State*>& particles,
	RandomStreams& streams, History& history) const {
	if (action_probs_.size() > 0) {
		return Random::GetCategory(action_probs_, Random::RANDOM.NextDouble());
	} else {
		return Random::RANDOM.NextInt(model_->NumActions());
	}
}

ValuedAction RandomPolicy::Search() {
	double dummy_value = Globals::NEG_INFTY;
	if (action_probs_.size() > 0) {
		return ValuedAction(
			Random::GetCategory(action_probs_, Random::RANDOM.NextDouble()),
			dummy_value);
	} else {
		return ValuedAction(Random::RANDOM.NextInt(model_->NumActions()),
			dummy_value);
	}
}

void RandomPolicy::Update(int action, OBS_TYPE obs) {
}

/* =============================================================================
 * MajorityActionPolicy class
 * =============================================================================*/

MajorityActionPolicy::MajorityActionPolicy(const DSPOMDP* model,
	const StatePolicy& policy, ParticleLowerBound* bound, Belief* belief) :
	Policy(model, bound, belief),
	policy_(policy) {
}

int MajorityActionPolicy::Action(const vector<State*>& particles,
	RandomStreams& streams, History& history) const {
	vector<double> frequencies(model_->NumActions());

	for (int i = 0; i < particles.size(); i++) {
		State* particle = particles[i];
		int action = policy_.GetAction(*particle);
		frequencies[action] += particle->weight;
	}

	int bestAction = 0;
	double bestWeight = frequencies[0];
	for (int a = 1; a < frequencies.size(); a++) {
		if (bestWeight < frequencies[a]) {
			bestWeight = frequencies[a];
			bestAction = a;
		}
	}

	return bestAction;
}

/* =============================================================================
 * ModeStatePolicy class
 * =============================================================================*/

ModeStatePolicy::ModeStatePolicy(const DSPOMDP* model,
	const StateIndexer& indexer, const StatePolicy& policy,
	ParticleLowerBound* bound, Belief* belief) :
	Policy(model, bound, belief),
	indexer_(indexer),
	policy_(policy) {
	state_probs_.resize(indexer_.NumStates());
}

int ModeStatePolicy::Action(const vector<State*>& particles,
	RandomStreams& streams, History& history) const {
	double maxWeight = 0;
	State* mode = NULL;
	for (int i = 0; i < particles.size(); i++) {
		State* particle = particles[i];
		int id = indexer_.GetIndex(particle);
		state_probs_[id] += particle->weight;

		if (state_probs_[id] > maxWeight) {
			maxWeight = state_probs_[id];
			mode = particle;
		}
	}

	for (int i = 0; i < particles.size(); i++) {
		state_probs_[indexer_.GetIndex(particles[i])] = 0;
	}

	assert(mode != NULL);
	return policy_.GetAction(*mode);
}

/* =============================================================================
 * MMAPStatePolicy class
 * =============================================================================*/

MMAPStatePolicy::MMAPStatePolicy(const DSPOMDP* model,
	const MMAPInferencer& inferencer, const StatePolicy& policy,
	ParticleLowerBound* bound, Belief* belief) :
	Policy(model, bound, belief),
	inferencer_(inferencer),
	policy_(policy) {
}

int MMAPStatePolicy::Action(const vector<State*>& particles,
	RandomStreams& streams, History& history) const {
	return policy_.GetAction(*inferencer_.GetMMAP(particles));
}

} // namespace despot
