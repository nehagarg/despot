/* 
 * File:   DespotWithBeliefTracking.cpp
 * Author: neha
 * 
 * Created on April 23, 2018, 1:32 PM
 */

#include <despot/solver/DespotWithBeliefTracking.h>
#include <despot/util/logging.h>
#include <map>

namespace despot {
DespotWithBeliefTracking::DespotWithBeliefTracking(const DSPOMDP* model, 
        ScenarioLowerBound* lb, 
        ScenarioUpperBound* ub, 
        Belief* belief): DESPOT(model, lb, ub, belief)
{
    std::cout << "Despot With belief tracking" << std::endl;
    o_helper_ = new DespotStaticFunctionOverrideHelperForBeliefTracking();
    o_helper_->solver_pointer = this;
}

void DespotStaticFunctionOverrideHelperForBeliefTracking::Expand(QNode* qnode, ScenarioLowerBound* lb,
	ScenarioUpperBound* ub, const DSPOMDP* model,
	RandomStreams& streams,
	History& history,  ScenarioLowerBound* learned_lower_bound, 
        SearchStatistics* statistics, 
        DespotStaticFunctionOverrideHelper* o_helper)
{
    //std::cout << "Expanding in Despot With belief tracking" << std::endl;
        VNode* parent = qnode->parent();
	streams.position(parent->depth());
	std::map<OBS_TYPE, VNode*>& children = qnode->children();

	const std::vector<State*>& particles = parent->particles();
        int observation_particle_size = parent->observation_particle_size;
                

	double step_reward = 0;

	// Partition particles by observation
	std::map<OBS_TYPE, std::vector<State*> > partitions;
        std::map<OBS_TYPE, std::vector<State*> > partitions_belief_; //Stores belief particles
        
	OBS_TYPE obs;
	double reward;
        int  num_particles_pushed = 0;
	for (int i = 0; i < particles.size(); i++) {
		State* particle = particles[i];
		logd << " Original: " << *particle << std::endl;

		State* copy = model->Copy(particle);

		logd << " Before step: " << *copy << std::endl;

		bool terminal = model->Step(*copy, streams.Entry(copy->scenario_id),
			qnode->edge(), reward, obs);

		step_reward += reward * copy->weight;

		logd << " After step: " << *copy << " " << (reward * copy->weight)
			<< " " << reward << " " << copy->weight << std::endl;

		if (!terminal) {
                    if(i< observation_particle_size)
                    {
                        if(partitions.count(obs) == 0)
                        {
                            for (std::map<OBS_TYPE, std::vector<State*> >::iterator it = partitions.begin();
		it != partitions.end(); it++)
                            {
                                OBS_TYPE obs_key = it->first;
                                for (int j = 0; j< partitions[obs_key].size();j++)
                                {
                                    State* copy1 = model->Copy(partitions[obs_key][j]);
                                    partitions_belief_[obs].push_back(copy1);
                                }
                            }
                        }
			partitions[obs].push_back(copy);
                        num_particles_pushed++;
                        
                    }
                    
                        for (std::map<OBS_TYPE, std::vector<State*> >::iterator it = partitions.begin();
		it != partitions.end(); it++) {
                            OBS_TYPE obs_key = it->first;
                            if(i>=observation_particle_size || obs_key!=obs)
                            {
                                State* copy1 = model->Copy(copy);
                                partitions_belief_[obs_key].push_back(copy1);
                            }
                            
                        }
                    if(i>= observation_particle_size)
                {
                    model->Free(copy);
                }
                    
		} 
                else {
			model->Free(copy);
		}
                
               
	}
        
            
        
        
        
           
	step_reward = Globals::Discount(parent->depth()) * step_reward
		- Globals::config.pruning_constant;//pruning_constant is used for regularization
        
        //std::cout << "Step reward = " << step_reward << std::endl;
	double lower_bound = step_reward;
	double upper_bound = step_reward;

	// Create new belief nodes
	for (std::map<OBS_TYPE, std::vector<State*> >::iterator it = partitions.begin();
		it != partitions.end(); it++) {
		OBS_TYPE obs = it->first;
                int observation_particle_size_ = partitions[obs].size();
                partitions[obs].insert(partitions[obs].end(),partitions_belief_[obs].begin(),partitions_belief_[obs].end());
		double total_weight = 0;
                for(int i = 0; i < partitions[obs].size();i++)
                {
                    double prob = model->ObsProb(obs, *partitions[obs][i], qnode->edge());
                //std::cout << "Obs Prob:" <<  prob << std::endl;
                
		 // Terminal state is not required to be explicitly represented and may not have any observation
			partitions[obs][i]->weight *= prob;
			total_weight += partitions[obs][i]->weight;
                        //Total weight should not be zero as one particle actually produced that observation
                }
                for(int i = 0; i < partitions[obs].size();i++)
                {
                    partitions[obs][i]->weight = partitions[obs][i]->weight/total_weight;
                }
                
                logd << " Creating node for obs " << obs << std::endl;
		VNode* vnode = new VNode(partitions[obs], parent->depth() + 1,
			qnode, obs);
                vnode->observation_particle_size = observation_particle_size_;
		logd << " New node created!" << std::endl;
		children[obs] = vnode;

		history.Add(qnode->edge(), obs);
		DESPOT::InitBounds(vnode, lb, ub, streams, history, learned_lower_bound, statistics, o_helper);
		history.RemoveLast();
		logd << " New node's bounds: (" << vnode->lower_bound() << ", "
			<< vnode->upper_bound() << ")" << std::endl;

		lower_bound += vnode->lower_bound()*observation_particle_size_/observation_particle_size;
		upper_bound += vnode->upper_bound()*observation_particle_size_/observation_particle_size;
	}
        //std::cout << "Upper bound = " << upper_bound - step_reward<< " Num particles pushed " << num_particles_pushed << std::endl;
	qnode->step_reward = step_reward;
	qnode->lower_bound(lower_bound);
	qnode->upper_bound(upper_bound);
	qnode->utility_upper_bound = upper_bound + Globals::config.pruning_constant;

	qnode->default_value = lower_bound; // for debugging
}

void DespotStaticFunctionOverrideHelperForBeliefTracking::Update(QNode* qnode)
{
    //std::cout << "Updating in tracking belief" << std::endl;
        double lower = qnode->step_reward;
	double upper = qnode->step_reward;
	double utility_upper = qnode->step_reward
		+ Globals::config.pruning_constant;

        double belief_lower = 0;
        double belief_upper = 0;
        double belief_utility_upper = 0;
        int total_particles = 0;
	std::map<OBS_TYPE, VNode*>& children = qnode->children();
	for (std::map<OBS_TYPE, VNode*>::iterator it = children.begin();
		it != children.end(); it++) {
		VNode* vnode = it->second;
                total_particles = total_particles + vnode->observation_particle_size;
		belief_lower += vnode->lower_bound()*vnode->observation_particle_size;
		belief_upper += vnode->upper_bound()*vnode->observation_particle_size;
		belief_utility_upper += vnode->utility_upper_bound*vnode->observation_particle_size;
	}
        lower += belief_lower/(total_particles*1.0);
        upper += belief_upper/(total_particles*1.0);
        utility_upper += belief_utility_upper/(total_particles*1.0);
        
	if (lower > qnode->lower_bound()) {
		qnode->lower_bound(lower);
	}
	if (upper < qnode->upper_bound()) {
		qnode->upper_bound(upper);
	}
	if (utility_upper < qnode->utility_upper_bound) {
		qnode->utility_upper_bound = utility_upper;
	}
}

int DespotStaticFunctionOverrideHelperForBeliefTracking::GetObservationParticleSize(VNode* vnode)
    {
        return vnode->observation_particle_size;
    }

void DespotWithBeliefTracking::CoreSearch(std::vector<State*> particles, RandomStreams& streams) {
    //std::cout << "Initiallizing contruct tree with learned policy solver ##################" << std::endl;
    //DESPOT::CoreSearch(particles, streams);

    //if(history_.Size()>= 28)
    //{
    
    //((LearningModel*)model_)->SetStoreObsHash(true);
    root_ = ConstructTree(particles, streams, lower_bound_, upper_bound_,
		model_, history_, Globals::config.time_per_move, statistics_, NULL, o_helper_);
    //((LearningModel*)model_)->SetStoreObsHash(false);
    
    //}
    //else
    //{
    //    DESPOT::CoreSearch(particles, streams);
    // }
}
}