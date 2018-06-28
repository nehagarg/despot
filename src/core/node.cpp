#include <despot/core/node.h>
#include <despot/solver/despot.h>
#include <pm_config.h>

using namespace std;

namespace despot {

/* =============================================================================
 * VNode class
 * =============================================================================*/

VNode::VNode(vector<State*>& particles, int depth, QNode* parent,
	OBS_TYPE edge) :
	belief_(NULL),
	depth_(depth),
	parent_(parent),
	edge_(edge),
	vstar(this),
	likelihood(1),
        rnn_state(NULL),
        rnn_output(NULL){
    
    particle_node_ = NULL;
    if(parent != NULL)
    {
        particle_node_ = parent->parent()->particle_node_->Child(parent->edge());
    }
    
    
    if(particle_node_ == NULL)
    {
        if(parent != NULL)
        {
            particle_node_ = new ParticleNode(particles, depth, parent->parent()->particle_node_, parent->edge());
            parent->parent()->particle_node_->Child(parent->edge(), particle_node_);
            
        }
        else
        {
            particle_node_ = new ParticleNode(particles, depth);
        }
    }

    logd << "Constructed vnode with " << particles.size() << " particles"
		<< endl;
	for (int i = 0; i < particles.size(); i++) {
		logd << " " << i << " = " << *(particles[i]) << endl;
                //particle_weight_[particles[i]->scenario_id] = (particles[i]->Weight());
                obs_particle_id_.push_back(particles[i]->scenario_id);
	}
        
        observation_particle_size = -1;
    }

    VNode::VNode(ParticleNode* particle_node, std::vector<int>& obs_particle_id, std::vector<double>& particle_weight, int obs_particle_size, int depth, QNode* parent, OBS_TYPE edge):
    particle_node_(particle_node),
            obs_particle_id_(obs_particle_id),
            particle_weight_(particle_weight),
            observation_particle_size(obs_particle_size),
        belief_(NULL),
	depth_(depth),
	parent_(parent),
	edge_(edge),
	vstar(this),
	likelihood(1),
        rnn_state(NULL),
        rnn_output(NULL){
        logd << "Constructed vnode with " << obs_particle_id_.size() << "/" << particle_node->particles_size()<< " particles"
		<< endl;
	//for (int i = 0; i < particle_node_->particles_.size(); i++)
        /*for (map<int, State*>::iterator it = particle_node_->particles_.begin();
                    it != particle_node_->particles_.end(); it++){
		logd << " " << it->first << " = " << *(it->second) << endl;
	}*/
    }


VNode::VNode(Belief* belief, int depth, QNode* parent, OBS_TYPE edge) :
	belief_(belief),
	depth_(depth),
	parent_(parent),
	edge_(edge),
	vstar(this),
	likelihood(1),
        rnn_state(NULL),
        rnn_output(NULL){
}

VNode::VNode(int count, double value, int depth, QNode* parent, OBS_TYPE edge) :
	belief_(NULL),
	depth_(depth),
	parent_(parent),
	edge_(edge),
	count_(count),
	value_(value),
        rnn_state(NULL),
        rnn_output(NULL){
}

VNode::~VNode() {
	for (int a = 0; a < children_.size(); a++) {
		QNode* child = children_[a];
		assert(child != NULL);
		delete child;
	}
	children_.clear();
        obs_particle_id_.clear();
        particle_weight_.clear();
	if (belief_ != NULL)
		delete belief_;
        if(rnn_state !=NULL)
        {
            Py_DECREF(rnn_state);
        }
        if(rnn_output != NULL)
        {
            Py_DECREF(rnn_output);
        }
        //delete(particle_node_);
}

Belief* VNode::belief() const {
	return belief_;
}
/*const map<int ,State*>& VNode::particles() const {
            return particle_node_->particles_;
        }
     */

    int VNode::particles_size() const {
        if(observation_particle_size > 0)
        {
            return particle_node_->particles_size();
        }
        else
        {
            return obs_particle_id_.size();
        }
    }

    const std::vector<int>& VNode::particle_ids() const {
        return obs_particle_id_;
    }


void VNode::depth(int d) {
	depth_ = d;
}

int VNode::depth() const {
	return depth_;
}

void VNode::parent(QNode* parent) {
	parent_ = parent;
}

QNode* VNode::parent() {
	return parent_;
}

OBS_TYPE VNode::edge() {
	return edge_;
}

double VNode::Weight() const {
    //double weight = 0;
    //for (int i = 0; i < particle_node_->particles_.size(); i++)
    //for (map<int, double>::const_iterator it = particle_weight_.begin();
    //                it != particle_weight_.end(); it++)
    //        weight += it->second;
    //return weight;
    return State::Weight(particle_node_,particle_weight_,obs_particle_id_,observation_particle_size);
    }

const vector<QNode*>& VNode::children() const {
	return children_;
}

vector<QNode*>& VNode::children() {
	return children_;
}

const QNode* VNode::Child(int action) const {
	return children_[action];
}

QNode* VNode::Child(int action) {
	return children_[action];
}

int VNode::Size() const {
	int size = 1;
	for (int a = 0; a < children_.size(); a++) {
		size += children_[a]->Size();
	}
	return size;
}

int VNode::PolicyTreeSize() const {
	if (children_.size() == 0)
		return 0;

	QNode* best = NULL;
	for (int a = 0; a < children_.size(); a++) {
		QNode* child = children_[a];
		if (best == NULL || child->lower_bound() > best->lower_bound())
			best = child;
	}
	return best->PolicyTreeSize();
}

void VNode::default_move(ValuedAction move) {
	default_move_ = move;
}

ValuedAction VNode::default_move() const {
	return default_move_;
}

void VNode::lower_bound(double value) {
	lower_bound_ = value;
}

double VNode::lower_bound() const {
	return lower_bound_;
}

void VNode::upper_bound(double value) {
	upper_bound_ = value;
}

double VNode::upper_bound() const {
	return upper_bound_;
}

bool VNode::IsLeaf() {
	return children_.size() == 0;
}

void VNode::Add(double val) {
	value_ = (value_ * count_ + val) / (count_ + 1);
	count_++;
}

void VNode::count(int c) {
	count_ = c;
}
int VNode::count() const {
	return count_;
}
void VNode::value(double v) {
	value_ = v;
}
double VNode::value() const {
	return value_;
}

void VNode::Free(const DSPOMDP& model) {
    particle_node_->Free(model);
    /*for (int i = 0; i < particle_node_->particles_.size(); i++) {
		model.Free(particle_node_->particles_[i]);
	}

	for (int a = 0; a < children().size(); a++) {
		QNode* qnode = Child(a);
		map<OBS_TYPE, VNode*>& children = qnode->children();
		for (map<OBS_TYPE, VNode*>::iterator it = children.begin();
			it != children.end(); it++) {
			it->second->Free(model);
		}
	}*/

}

std::ostream& operator<<(std::ostream& os, const VNode& vnode)
{
    std::set<int> obs_particle_ids;
        for (int i = 0; i < vnode.obs_particle_id_.size(); i++) {
            /*if(i==this->observation_particle_size)
            {
                os << "||||";
            }*/
            if(vnode.observation_particle_size > 0)
            {
                obs_particle_ids.insert(vnode.obs_particle_id_[i]);
            }
            
            //((vnode.particle_node_)->particles_[vnode.obs_particle_id_[i]])->weight = vnode.particle_weight_[vnode.obs_particle_id_[i]];
            os << " " << i << " = ";// << *((vnode.particle_node_->particles_)[vnode.obs_particle_id_[i]]) << endl;
            os << "(state_id = " << (vnode.particle_node_->particle(vnode.obs_particle_id_[i]))->state_id << ", weight = " << (vnode.particle_node_->particle(vnode.obs_particle_id_[i]))->Weight(vnode.particle_weight_)
		<< ", text = " << (vnode.particle_node_->particle(vnode.obs_particle_id_[i]))->text() << ")";
                os << endl;
        }
        if(vnode.observation_particle_size > 0)
        {
            os << "||||";
            
        
            //for (map<int, State*>::iterator it = vnode.particle_node_->particles_.begin();
            //        it != vnode.particle_node_->particles_.end(); it++)
            std::vector<State*> particles;
            ParticleNode::particles_vector(vnode.particle_node_,vnode.obs_particle_id_, vnode.observation_particle_size, particles, false);
            for(int i = 0; i < particles.size(); i++)
            {
                
                
                   // it->second->weight = vnode.particle_weight_[it->first];
                    os << " " << particles[i]->scenario_id << " = " ; //<<*(it->second) << endl;
                    os << "(state_id = " << particles[i]->state_id << ", weight = " << particles[i]->Weight(vnode.particle_weight_)
		<< ", text = " << particles[i]->text() << ")";
                    os << endl; 
                
            }
        }
}

void VNode::PrintPolicyTree(int depth, ostream& os) {
	if (depth != -1 && this->depth() > depth)
		return;
        
        os << "(" << "d:" << this->default_move().value <<
		" l:" << this->lower_bound() << ", u:" << this->upper_bound()
		<< ", w:" << this->Weight() << ", weu:" << DESPOT::WEU(this)
		<< ")";
        
        //os<<*(this);
		
		os << endl;

	vector<QNode*>& qnodes = children();
	if (qnodes.size() == 0) {
		int astar = this->default_move().action;
		os << repeat("|   ", this->depth()) << this << "-a=" << astar << endl;
	} else {
		QNode* qstar = NULL;
		for (int a = 0; a < qnodes.size(); a++) {
			QNode* qnode = qnodes[a];
			if (qstar == NULL || qnode->lower_bound() > qstar->lower_bound()) {
				qstar = qnode;
			}
		}

		os << repeat("|   ", this->depth()) <<this << "-a=" << qstar->edge() << ": " << "(d:" << qstar->default_value << ", l:" << qstar->lower_bound()
			<< ", u:" << qstar->upper_bound()
			<< ", r:" << qstar->step_reward << ")"<< endl;

		vector<OBS_TYPE> labels;
		map<OBS_TYPE, VNode*>& vnodes = qstar->children();
		for (map<OBS_TYPE, VNode*>::iterator it = vnodes.begin();
			it != vnodes.end(); it++) {
			labels.push_back(it->first);
		}

		for (int i = 0; i < labels.size(); i++) {
			if (depth == -1 || this->depth() + 1 <= depth) {
				os << repeat("|   ", this->depth()) << "| o=" << labels[i]
					<< ": ";
				qstar->Child(labels[i])->PrintPolicyTree(depth, os);
			}
		}
	}
}


void VNode::PrintTree(int depth, ostream& os) {
	if (depth != -1 && this->depth() > depth)
		return;

	if (this->depth() == 0) {
		os << "d - default value" << endl
			<< "l - lower bound" << endl
			<< "u - upper bound" << endl
			<< "r - totol weighted one step reward" << endl
			<< "w - total particle weight" << endl;
	}

	os << "(" << "d:" << this->default_move().value <<
		" l:" << this->lower_bound() << ", u:" << this->upper_bound()
		<< ", w:" << this->Weight() << ", weu:" << DESPOT::WEU(this)
		<< ")";
        /*for (int i = 0; i < this->particle_node_->particles_.size(); i++) {
            if(i==this->observation_particle_size)
            {
                os << "||||";
            }
            os << " " << i << " = " << *((this->particle_node_->particles_)[i]) << endl;
        }
		
		os << endl;
         */ 
        //os << *(this);
        os << endl;
	vector<QNode*>& qnodes = children();
	for (int a = 0; a < qnodes.size(); a++) {
		QNode* qnode = qnodes[a];

		vector<OBS_TYPE> labels;
		map<OBS_TYPE, VNode*>& vnodes = qnode->children();
		for (map<OBS_TYPE, VNode*>::iterator it = vnodes.begin();
			it != vnodes.end(); it++) {
			labels.push_back(it->first);
		}

		os << repeat("|   ", this->depth()) << "a="
			<< qnode->edge() << ": "
			<< "(d:" << qnode->default_value << ", l:" << qnode->lower_bound()
			<< ", u:" << qnode->upper_bound()
			<< ", r:" << qnode->step_reward << ")" << endl;

		for (int i = 0; i < labels.size(); i++) {
			if (depth == -1 || this->depth() + 1 <= depth) {
				os << repeat("|   ", this->depth()) << "| o=" << labels[i]
					<< ": ";
				qnode->Child(labels[i])->PrintTree(depth, os);
			}
		}
	}
}

/* =============================================================================
 * QNode class
 * =============================================================================*/

QNode::QNode(VNode* parent, int edge) :
	parent_(parent),
	edge_(edge),
	vstar(NULL) {
}

QNode::QNode(int count, double value) :
	count_(count),
	value_(value) {
}

QNode::~QNode() {
	for (map<OBS_TYPE, VNode*>::iterator it = children_.begin();
		it != children_.end(); it++) {
		assert(it->second != NULL);
		delete it->second;
	}
	children_.clear();
}

void QNode::parent(VNode* parent) {
	parent_ = parent;
}

VNode* QNode::parent() {
	return parent_;
}

int QNode::edge() {
	return edge_;
}

map<OBS_TYPE, VNode*>& QNode::children() {
	return children_;
}

VNode* QNode::Child(OBS_TYPE obs) {
	return children_[obs];
}

int QNode::Size() const {
	int size = 0;
	for (map<OBS_TYPE, VNode*>::const_iterator it = children_.begin();
		it != children_.end(); it++) {
		size += it->second->Size();
	}
	return size;
}

int QNode::PolicyTreeSize() const {
	int size = 0;
	for (map<OBS_TYPE, VNode*>::const_iterator it = children_.begin();
		it != children_.end(); it++) {
		size += it->second->PolicyTreeSize();
	}
	return 1 + size;
}

double QNode::Weight() const {
	double weight = 0;
	for (map<OBS_TYPE, VNode*>::const_iterator it = children_.begin();
		it != children_.end(); it++) {
		weight += it->second->Weight();
	}
	return weight;
}

void QNode::lower_bound(double value) {
	lower_bound_ = value;
}

double QNode::lower_bound() const {
	return lower_bound_;
}

void QNode::upper_bound(double value) {
	upper_bound_ = value;
}

double QNode::upper_bound() const {
	return upper_bound_;
}

void QNode::Add(double val) {
	value_ = (value_ * count_ + val) / (count_ + 1);
	count_++;
}

void QNode::count(int c) {
	count_ = c;
}

int QNode::count() const {
	return count_;
}

void QNode::value(double v) {
	value_ = v;
}

double QNode::value() const {
	return value_;
    }

ParticleNode::ParticleNode(std::vector<State*>& particles, int depth, ParticleNode* parent, int edge)
    :depth_(depth),parent_(parent),edge_(edge)
    {
    obs_.resize(Globals::config.num_scenarios, Globals::NEG_INFTY);
    reward_.resize(Globals::config.num_scenarios,Globals::NEG_INFTY);
    terminal_.resize(Globals::config.num_scenarios, FALSE);
    //std::cout<< DESPOT::NumActions << "\n";
    children_.resize(DESPOT::NumActions, NULL);
    particle_obs_prob_.resize(Globals::config.num_scenarios);
    particles_.resize(Globals::config.num_scenarios, NULL);
    num_particles = particles.size();
        for(int i = 0; i < particles.size(); i++)
        {
            //std::map<OBS_TYPE, double> m;
            //particle_obs_prob_[particles[i]->scenario_id] = m;
            particles_[particles[i]->scenario_id] = particles[i];
        }
    }

//const std::map<int,State*>& ParticleNode::particles() const {
//    return particles_ ;
//    }

    State* ParticleNode::particle(int i) const {
       // try
        //{
            return particles_[i];
        //}
        /*catch(const std::out_of_range& e)
        {
            return NULL;
        }*/
    }
   
    const OBS_TYPE ParticleNode::obs(int i) const {
        return obs_[i];
    }
        const double ParticleNode::reward(int i) const {
            return reward_[i];
    }

    const bool ParticleNode::terminal(int i) const {
        //try
        //{
            return terminal_[i];
        //}
        /*catch(const std::out_of_range& e)
        {
            return FALSE;
        }*/
    }
    double ParticleNode::obs_prob(int i, OBS_TYPE o) {
        //try
        //{
            return particle_obs_prob_[i][o].GetVal();
            //return ans;
        //}
        //catch(const std::out_of_range& e)
        //{
        //    return -1;
        //}
    }



ParticleNode* ParticleNode::Child(int action) {
    

            ParticleNode* ans =  children_[action];
            return ans;
      /*/
        catch(const std::out_of_range& e)
        {
            return NULL;
        }
    */
    }

    void ParticleNode::Child(int action, ParticleNode* p) {
        children_[action] = p;
    }


    void ParticleNode::depth(int d) {
        depth_ = d;
    }

    int ParticleNode::depth() const {
        return depth_;
    }
    
    
    ParticleNode* ParticleNode::parent() {
            return parent_;
    }
    
    void ParticleNode::parent(ParticleNode* parent) {
                parent_ = parent;
    }
        int ParticleNode::particles_size() const {
            return num_particles;
    }

    void ParticleNode::Add(State* particle, OBS_TYPE obs, double reward, bool terminal) {
        particles_[particle->scenario_id] = particle;
        obs_[particle->scenario_id] = obs;
        reward_[particle->scenario_id] = reward;
        terminal_[particle->scenario_id] = terminal;
        //std::map<OBS_TYPE, double> m;
        //particle_obs_prob_[particle->scenario_id] = m;
        num_particles++;
    }
        void ParticleNode::AddObsProb(int i, OBS_TYPE o, double prob) {
            particle_obs_prob_[i][o].SetVal(prob);
    }


    void ParticleNode::Free(const DSPOMDP& model) {
            //for (map<int, State*>::iterator it = particles_.begin();
            //        it != particles_.end(); it++) {
        //std::cout<< "Freeing particles" << std::endl;
        //std::cout<< "Particles before freeing" << model.NumActiveParticles() << std::endl;
        for(int i = 0; i < particles_.size(); i++)
            {
            if(particles_[i] != NULL)
            {
		model.Free(particles_[i]);
                
            }
	}
          //std::cout<< "Particles after freeing" << model.NumActiveParticles() << std::endl;
          //  particles_.clear();
          //std::cout << "Children " << children_.size() << std::endl;

            //for (map<int, ParticleNode*>::iterator it = children_.begin();
            //        it != children_.end(); it++) {
            for(int i = 0; i < children_.size(); i++)
            {
                if(children_[i] != NULL)
                {
                    children_[i]->Free(model);
                }
                /*else
                {
                    std::cout << "Children " << i << "is null" << std::endl;
                }*/
                   // it->second->Free(model);
            }
    }

    void ParticleNode::particles_vector(const ParticleNode* particle_node, const std::vector<int>& obs_particle_ids, const int observation_particle_size, std::vector<State*>& particle_vector, bool get_all) {
        //vector<State*> particles_vector ;
                
        if(observation_particle_size > 0)
        {
            //for (map<int,State*>::iterator it = particle_node->particles_.begin();
            //it != particle_node->particles_.end(); it++)
            int obs_particle_index = 0;
            
             for(int i = 0; i < particle_node->particles_.size(); i++)
            {
                 if(!get_all && i==obs_particle_ids[obs_particle_index])
                 {
                     obs_particle_index++;
                 }
                 else
                 {
                    if(!particle_node->terminal(i))
                    {
                        State* particle =  particle_node->particle(i);
                        if(particle != NULL)
                        {
                            particle_vector.push_back(particle);
                        }
                    }
                 }
            }
        }
        else
        {
            for(int i = 0; i < obs_particle_ids.size(); i++)
            {
             int ii = obs_particle_ids[i];
             State* particle = particle_node->particle(ii);
             particle_vector.push_back(particle);
            } 
        }
        
        //return particles_vector;
    }
        ParticleNode::~ParticleNode() {
        obs_.clear();
        reward_.clear();
        terminal_.clear();
        particle_obs_prob_.clear();
        for (int a = 0; a < children_.size(); a++) {
		ParticleNode* child = children_[a];
		if(child != NULL)
                {
                    delete child;
                }
	}
        children_.clear();
        particles_.clear();
    }




} // namespace despot
