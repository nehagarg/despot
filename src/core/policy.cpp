#include <despot/core/policy.h>
#include <despot/core/pomdp.h>
#include <unistd.h>

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

ValuedAction Policy::Value(ParticleNode* particle_node, std::vector<double>& particle_weights, std::vector<int> & obs_particle_ids,
	RandomStreams& streams, History& history, int observation_particle_size) const {
    //std::cout<< "Observation particle size" << observation_particle_size << std::endl;
	//vector<State*> copy;
        //vector<double> copy_weight;
        /*if(observation_particle_size > 0)
        {
            for (int i = 0; i < particles.size(); i++)
            copy_weight.push_back(particles[i]->weight);
        }*/
        //std::cout << "Num active particles 0 :" << model_->NumActiveParticles() << std::endl;
	//for (int i = 0; i < particles.size(); i++)
	//	copy.push_back(model_->Copy(particles[i]));
        
        //std::cout << "Num active particles 1 :" << model_->NumActiveParticles() << std::endl;
	initial_depth_ = history.Size();
	ValuedAction va = RecursiveValue(particle_node, particle_weights, obs_particle_ids, streams, history, observation_particle_size );
        
        //std::cout << "Num active particles 2 :" << model_->NumActiveParticles() << std::endl;
        //Belief particles get freed in recursive value function
        //int free_particle_size = copy.size();
        /*if(observation_particle_size > 0)
        {
            free_particle_size = observation_particle_size;
        }*/
        //std::cout<< "Free particle size" << free_particle_size << std::endl;
        /*if(observation_particle_size < 0)
        {
	for (int i = 0; i < free_particle_size; i++)
		model_->Free(copy[i]);
        }*/
        
        //std::cout << "Num active particles 3 :" << model_->NumActiveParticles() << std::endl;
        //std::cout<< "Valued action" << va << std::endl;

	return va;
}

ValuedAction Policy::RecursiveValue(ParticleNode* particle_node, std::vector<double>& particle_weights,
	std::vector<int> & obs_particle_ids, RandomStreams& streams, History& history, 
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
            ValuedAction ans = particle_lower_bound_->Value(particle_node, particle_weights, obs_particle_ids, obs_particle_size);
            /*if(obs_particle_size > 0)
            {
                for (int i = 0; i < particles.size(); i++)
                {
                    model_->Free(particles[i]);
                }
                    
            }*/
            return ans;
            
            
		
	} else {
		int action = Action(particle_node, particle_weights, obs_particle_ids, streams, history, obs_particle_size);
                if(action < 0)
                {   
                   ValuedAction ans =  particle_lower_bound_->Value(particle_node,particle_weights, obs_particle_ids, obs_particle_size);
                    /*if(obs_particle_size > 0)
                    {
                        for (int i = 0; i < particles.size(); i++)
                        {
                            model_->Free(particles[i]);
                        }

                    }*/
                    return ans;
                    //return particle_lower_bound_->Value(particles);
                }
                
		double value = 0;
                int observation_particle_size = obs_particle_size;
                
                
                
                //std::map<OBS_TYPE, std::vector<State*> > partitions_belief_; //Stores belief particles
		map<OBS_TYPE, vector<int> > partitions;
                //map<OBS_TYPE, vector<double> > partitions_weight;
                //map<OBS_TYPE, vector<double> > partitions_belief_weight;
                ParticleNode* particle_node_child = particle_node->Child(action);
                if(particle_node_child == NULL)
                    {
                        std::vector<State*> temp;
                        particle_node_child = new ParticleNode(temp, particle_node->depth() + 1, particle_node, action);
                        particle_node->Child(action,particle_node_child);
                    }
		OBS_TYPE obs;
		double reward;
		//for (int i = 0; i < particles.size(); i++)
                //std::set<int> obs_particle_id_set;
                for (int i = 0; i < obs_particle_ids.size(); i++){
                   /* if(obs_particle_size > 0)
                    {
                        obs_particle_id_set.insert(obs_particle_ids[i]);
                    }*/
                    //State* particle = particle_node->particle(obs_particle_ids[i]);
                    
                    bool terminal;
                    if(particle_node_child->particle(obs_particle_ids[i]) == NULL )
                    {
			State* copy = model_->Copy(particle_node->particle(obs_particle_ids[i]));
			terminal = model_->Step(*copy,
				streams.Entry(obs_particle_ids[i]), action, reward, obs);
                        particle_node_child->Add(copy,obs,reward, terminal);
                    }
                    else
                    {
                        obs = particle_node_child->obs(obs_particle_ids[i]);
                        reward = particle_node_child->reward(obs_particle_ids[i]);
                        terminal = particle_node_child->terminal(obs_particle_ids[i]);
                        
                    }
                    /*double particle_weight;
                    if(obs_particle_size > 0)
                    {
                        particle_weight = particle_weights[obs_particle_ids[i]];
                    }
                    else
                    {
                       particle_weight =  particle_node->particle(obs_particle_ids[i])->weight;
                    }*/
                    value += reward * particle_node->particle(obs_particle_ids[i])->Weight(particle_weights);

			if (!terminal) {
                           
                           
                                partitions[obs].push_back(obs_particle_ids[i]);
                        }
                }
                
                if(obs_particle_size > 0)
                {
                    std::vector<State*> particles;
                    ParticleNode::particles_vector(particle_node, obs_particle_ids, obs_particle_size, particles, false);
                  //for (map<int, State*>::iterator it = particle_node->particles_.begin();
                   // it != particle_node->particles_.end(); it++){
                    for(int i = 0; i < particles.size(); i++)
                    {
                        State* particle = particles[i];
                      //if(obs_particle_id_set.find(it->first) == obs_particle_id_set.end())
                      //{
                          //Not checking for particle_child_node_null as it would have been created in previou loop
                          //if(!particle_node->terminal(it->first)){
                            bool terminal;
                            if(particle_node_child->particle(particles[i]->scenario_id) == NULL )
                            {
                                State* copy = model_->Copy(particle);
                                terminal = model_->Step(*copy,
                                        streams.Entry(particle->scenario_id), action, reward, obs);
                                particle_node_child->Add(copy,obs,reward, terminal);
                            }
                            else
                            {
                                obs = particle_node_child->obs(particle->scenario_id);
                                reward = particle_node_child->reward(particle->scenario_id);
                                terminal = particle_node_child->reward(particle->scenario_id);

                            }
                            value += reward * particle_weights[particle->scenario_id];

                          //}
                      //}
                  }
                    
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
		vector<State*> particles;
                if(obs_particle_size > 0 && partitions.size() > 0)
                {
                            ParticleNode::particles_vector(particle_node_child, obs_particle_ids, obs_particle_size, particles, true);
                }
                for (map<OBS_TYPE, vector<int> >::iterator it = partitions.begin();
			it != partitions.end(); it++) {
			OBS_TYPE obs = it->first;
                        
                        int observation_particle_size_ = obs_particle_size;
                        std::vector<double> particle_weights_;
                        if(obs_particle_size > 0)
                        {
                            particle_weights_.resize(Globals::config.num_scenarios, 0);
                            observation_particle_size_ = partitions[obs].size();
                            //partitions[obs].insert(partitions[obs].end(),partitions_belief_[obs].begin(),partitions_belief_[obs].end());
                            //partitions_weight[obs].insert(partitions_weight[obs].end(),
                            //        partitions_belief_weight[obs].begin(),
                            //        partitions_belief_weight[obs].end());
                            double total_weight = 0;
                            
                            //for (map<int, State*>::iterator itt = particle_node_child->particles_.begin();
                    //itt != particle_node_child->particles_.end(); itt++)
                            //for(int i = 0; i < partitions[obs].size();i++)
                            for(int i = 0; i < particles.size(); i++)
                            {
                                //if(!particle_node_child->terminal(itt->first))
                                //{
                                    double prob = particle_node_child->obs_prob(particles[i]->scenario_id, obs);
                                    //model_->PrintObs(*partitions[obs][i], obs);
                                    if(prob < 0)
                                    {
                                        prob = model_->ObsProb(obs, *(particles[i]), action);
                                        particle_node_child->AddObsProb(particles[i]->scenario_id, obs, prob);
                                    }
                                
                                //std::cout << "Obs Prob:" <<  prob << " " << *partitions[obs][i] << std::endl;
                
                                 // Terminal state is not required to be explicitly represented and may not have any observation
                                //partitions_weight[obs][i] *= prob;
                                //total_weight += partitions_weight[obs][i];
                                particle_weights_[particles[i]->scenario_id] = particle_weights[particles[i]->scenario_id] *prob;
                                total_weight += particle_weights_[particles[i]->scenario_id];
                                //}
                                //else
                                //{
                                //    particle_weights_[itt->first] = 0;
                                //}
                //Total we
                        //Total weight should not be zero as one particle actually produced that observation
                            }
                            for(int i = 0; i < particle_weights_.size();i++)
                            {
                                //partitions_weight[obs][i]= partitions_weight[obs][i]/total_weight;
                                particle_weights_[i] = particle_weights_[i]/total_weight;
                            }
                
                        }
                        //std::cout<< "Recursive Partitioned Observation size" << observation_particle_size_<< std::endl;
			history.Add(action, obs);
			streams.Advance();
			ValuedAction va = RecursiveValue(particle_node_child, particle_weights_, it->second, streams, history, observation_particle_size_);
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
        VNode *vnode = new VNode(particles);
	int action = Action(vnode->particle_node_,vnode->particle_weight_, vnode->obs_particle_id_ , streams, history_, particles.size());
	double dummy_value = Globals::NEG_INFTY;

	for (int i = 0; i < particles.size(); i++)
            vnode->Free(*model_);
            //model_->Free(particles[i]);

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

int BlindPolicy::Action(ParticleNode* particle_node, std::vector<double>& particle_weights, std::vector<int> & obs_particle_ids, RandomStreams& streams,
	History& history, int observation_particle_size) const {
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

int RandomPolicy::Action(ParticleNode* particle_node, std::vector<double>& particle_weights, std::vector<int> & obs_particle_ids,
	RandomStreams& streams, History& history, int observation_particle_size) const {
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

int MajorityActionPolicy::Action(ParticleNode* particle_node, std::vector<double>& particle_weights, std::vector<int> & obs_particle_ids,
	RandomStreams& streams, History& history, int observation_particle_size) const {
	vector<double> frequencies(model_->NumActions());

        std::vector<State*> particles;
        ParticleNode::particles_vector(particle_node, obs_particle_ids, observation_particle_size, particles);
        /*if(observation_particle_size > 0)
        {
            
            for (map<int, State*>::iterator it = particle_node->particles_.begin();
                    it != particle_node->particles_.end(); it++)
            {
                State* particle = it->second;
                double particle_weight = particle->Weight(particle_weights);
                int action = policy_.GetAction(*particle);
		frequencies[action] += particle_weight;
            }
        }
        else
        {
         for(int i = 0; i < obs_particle_ids.size(); i++)
            {
             int ii = obs_particle_ids[i];
             const State* particle = particle_node->particle(ii);
             double particle_weight = particle->Weight(particle_weights);
                
                int action = policy_.GetAction(*particle);
		frequencies[action] += particle_weight;
            }   
        
        }*/
	for (int i = 0; i < particles.size(); i++) {
		State* particle = particles[i];
		int action = policy_.GetAction(*particle);
		frequencies[action] += particle->Weight(particle_weights);
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

int ModeStatePolicy::Action(ParticleNode* particle_node, std::vector<double>& particle_weights, std::vector<int> & obs_particle_ids,
	RandomStreams& streams, History& history, int observation_particle_size) const {
	double maxWeight = 0;
	State* mode = NULL;
        std::vector<int> particle_ids;
        vector<State*> particles;
        ParticleNode::particles_vector(particle_node, obs_particle_ids, observation_particle_size, particles);
        /*
        if(observation_particle_size > 0)
        {
            
            for (map<int, State*>::iterator it = particle_node->particles_.begin();
                    it != particle_node->particles_.end(); it++)
            {
                State* particle = it->second;
                double particle_weight = particle->Weight(particle_weights);
                int id = indexer_.GetIndex(particle);
		state_probs_[id] += particle_weight;

		if (state_probs_[id] > maxWeight) {
			maxWeight = state_probs_[id];
			mode = particle;
		}
                particle_ids.push_back(id);
            }
        }
        else
        {
         for(int i = 0; i < obs_particle_ids.size(); i++)
            {
             int ii = obs_particle_ids[i];
             State* particle = particle_node->particle(ii);
             double particle_weight = particle->Weight(particle_weights);
                
                int id = indexer_.GetIndex(particle);
		state_probs_[id] += particle_weight;

		if (state_probs_[id] > maxWeight) {
			maxWeight = state_probs_[id];
			mode = particle;
		}
                particle_ids.push_back(id);
            }   
        
        }
        */
        
	for (int i = 0; i < particles.size(); i++) {
		State* particle = particles[i];
		int id = indexer_.GetIndex(particle);
		state_probs_[id] += particle->Weight(particle_weights);

		if (state_probs_[id] > maxWeight) {
			maxWeight = state_probs_[id];
			mode = particle;
		}
                 particle_ids.push_back(id);
	}

	for (int i = 0; i < particle_ids.size(); i++) {
		state_probs_[particle_ids[i]] = 0;
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

int MMAPStatePolicy::Action(ParticleNode* particle_node, std::vector<double>& particle_weights, std::vector<int> & obs_particle_ids,
	RandomStreams& streams, History& history, int observation_particle_size) const {
	return policy_.GetAction(*inferencer_.GetMMAP(particle_node,particle_weights,obs_particle_ids, observation_particle_size));
}

} // namespace despot
