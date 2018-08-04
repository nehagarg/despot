#include <despot/core/node.h>
#include <despot/solver/despot.h>

using namespace std;

namespace despot {

/* =============================================================================
 * VNode class
 * =============================================================================*/

VNode::VNode(vector<State*>& particles, int depth, QNode* parent,
	OBS_TYPE edge) :
	particles_(particles),
	belief_(NULL),
	depth_(depth),
	parent_(parent),
	edge_(edge),
	vstar(this),
	likelihood(1),
        rnn_state(NULL),
        rnn_output(NULL),
        extra_node(false),
        common_parent_(NULL),
        obs_probs_holder(NULL),
        has_estimated_upper_bound_value(false),
        count_(0){
    logd << "Constructed vnode with " << particles_.size() << " particles"
		<< endl;
	for (int i = 0; i < particles_.size(); i++) {
		logd << " " << i << " = " << *particles_[i] << endl;
	}
        observation_particle_size = -1;
        if(Globals::config.track_alpha_vector)
        {
            common_parent_ = new QNode(particles);
            particle_weights.resize(Globals::config.num_scenarios, 0);
            for (int i = 0; i < particles_.size(); i++) {
		particle_weights[particles_[i]->scenario_id] = particles_[i]->weight;
	}
            
           //upper_bound_alpha_vector_.resize(Globals::config.num_scenarios, 0);
           //lower_bound_alpha_vector.resize(Globals::config.num_scenarios, 0); 
        }
        else
            {
                common_parent_ = NULL;
            }
    }

    VNode::VNode(int depth, QNode* parent, OBS_TYPE edge):
    belief_(NULL),
	depth_(depth),
	parent_(parent),
	edge_(edge),
	vstar(this),
	likelihood(1),
        rnn_state(NULL),
            rnn_output(NULL),
            extra_node(false),
            obs_probs_holder(NULL),
            has_estimated_upper_bound_value(false),
            count_(0)
    {
        particle_weights.resize(Globals::config.num_scenarios, 0);
        //upper_bound_alpha_vector_.resize(Globals::config.num_scenarios, 0);
        //obs_probs.resize(Globals::config.num_scenarios, 0);
        observation_particle_size = -1;
    }
    VNode::VNode(int depth, QNode* parent, QNode* common_parent, OBS_TYPE edge):
    VNode(depth, parent, edge){
        common_parent_ = common_parent;
    }


VNode::VNode(Belief* belief, int depth, QNode* parent, OBS_TYPE edge) :
	belief_(belief),
	depth_(depth),
	parent_(parent),
	edge_(edge),
	vstar(this),
	likelihood(1),
        rnn_state(NULL),
        rnn_output(NULL),
        extra_node(false){
}

VNode::VNode(int count, double value, int depth, QNode* parent, OBS_TYPE edge) :
	belief_(NULL),
	depth_(depth),
	parent_(parent),
	edge_(edge),
	count_(count),
	value_(value),
        rnn_state(NULL),
        rnn_output(NULL),
        extra_node(false){
}

VNode::~VNode() { 
	for (int a = 0; a < children_.size(); a++) {
		QNode* child = children_[a];
		assert(child != NULL);
		delete child;
                }
	children_.clear();
        if(Globals::config.track_alpha_vector)
        {
        if(common_parent_->parent() == NULL)
        {
            FreeCommonParent(common_parent_);
        }
        }
        particle_weights.clear();
        //upper_bound_alpha_vector_.clear();
        lower_bound_alpha_vector_.clear();
        
	if (belief_ != NULL)
        {
		delete belief_;
        }
        if(rnn_state !=NULL)
        {
            Py_DECREF(rnn_state);
        }
        if(rnn_output != NULL)
        {
            Py_DECREF(rnn_output);
        }
}

Belief* VNode::belief() const {
	return belief_;
}

const vector<State*>& VNode::particles() const {
    if(Globals::config.track_alpha_vector)
    {
        return common_parent_->particles_;
    }
    else
    {
        return particles_;
    }
}

    double VNode::calculate_lower_bound() const {
        double lower_bound = 0;
        //const std::vector<State*>& particles = this->particles();
        //std::cout << "Lower bound alpha vector" << lower_bound_alpha_vector << std::endl;
        //std::cout << "particle_weight_size" << particle_weights.size() << std::endl;
        for(int i = 0; i < Globals::config.num_scenarios; i++)
        {
          //  std::cout << "Particle_weight " << particle_weights[i];
            //int particle_index = particles[i]->scenario_id;
            lower_bound += particle_weights[i]*((*(lower_bound_alpha_vector.value_array))[i]);
        }
        //std::cout << "Lower bound is " << lower_bound << std::endl;
        return lower_bound;
    }
    
    double VNode::calculate_upper_bound() const {
        double upper_bound = 0;
        //const std::vector<State*>& particles = this->particles();
        for(int i = 0; i < Globals::config.num_scenarios; i++)
        {
          //  std::cout << "Particle_weight " << particle_weights[i];
            //int particle_index = particles[i]->scenario_id;            
            upper_bound += particle_weights[i]* ((*(upper_bound_alpha_vector.value_array))[i]);
        }
        //std::cout << "Calculating upper bound " << upper_bound << upper_bound_alpha_vector << std::endl;
        return upper_bound;
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
    if(Globals::config.track_alpha_vector)
    {
        //Can simply return 1 as weight sums up to 1
        return 1.0;
        double w = 0;
        for(int i = 0 ; i < particle_weights.size(); i++)
        {
            w = w + particle_weights[i];
        }
        return w;
    }
    else
    {
	return State::Weight(particles_);
    }
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

void VNode::CreateCommonQNode(int action) {
    int common_children_size = common_parent_->common_children_.size();
    if(common_children_size==action)
    {
        QNode* common_qnode = new QNode(this, action);
        common_parent_->common_children_.push_back(common_qnode);
    }
    else
    {
        //assert(common_children_size == num_actions);
    }
    }

    QNode* VNode::common_parent() {
        return common_parent_;
    }


    const QNode* VNode::CommonChild(int action) const {
        return common_parent_->common_children_[action];
    }

    QNode* VNode::CommonChild(int action) {
        return common_parent_->common_children_[action];
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

    double VNode::estimated_upper_bound() const {
        if(has_estimated_upper_bound_value)
        {
            return estimated_upper_bound_;
        }
        else
        {
            return upper_bound_;
        }
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
void VNode::FreeCommonParent(QNode* p)
{
    if(p->common_children_.size() == 0)
    {
        delete p;
    }
    else
    {
        for (int a = 0; a < p->common_children_.size(); a++)
        {
            FreeCommonParent(p->common_children_[a]);
        }
        delete p;
    }
}
void VNode::Free(const DSPOMDP& model) {
	for (int i = 0; i < particles_.size(); i++) {
		model.Free(particles_[i]);
	}
        if(Globals::config.track_alpha_vector)
        {
        if(common_parent_->default_move.value_array != NULL)
        {
            delete common_parent_->default_move.value_array;
            common_parent_->default_move.value_array = NULL;
        }
        }
        /*if(lower_bound_alpha_vector.value_array != NULL)
        {
            lower_bound_alpha_vector.value_array->clear();
        }*/

	for (int a = 0; a < children().size(); a++) {
		QNode* qnode = Child(a);
                for (int i = 0; i < qnode->particles_.size(); i++) {
                    model.Free(qnode->particles_[i]);
                }
                if (qnode->default_move.value_array != NULL)
                {
                    //This one is always created using new
                    delete qnode->default_move.value_array;
                }
		map<OBS_TYPE, VNode*>& children = qnode->children();
		for (map<OBS_TYPE, VNode*>::iterator it = children.begin();
			it != children.end(); it++) {
			it->second->Free(model);
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
        for (int i = 0; i < this->particles_.size(); i++) {
            if(i==this->observation_particle_size)
            {
                os << "||||";
            }
            os << " " << i << " = " << *((this->particles_)[i]) << endl;
        }
		
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
			<< "u - upper bound" << endl;
                if(Globals::config.estimate_upper_bound)
                {
                    os << "eu - estimated upper bound" << endl;
                }
			os << "r - totol weighted one step reward" << endl
			<< "w - total particle weight" << endl;
	}

	os << "(" << "d:" << this->default_move().value <<
		" l:" << this->lower_bound() << ", u:" << this->upper_bound();
        if(Globals::config.estimate_upper_bound)
                {
                    os << ", eu" << this->estimated_upper_bound();
                }
		os << ", w:" << this->Weight() << ", weu:" << DESPOT::WEU(this)
		<< ")";
        for (int i = 0; i < this->particles_.size(); i++) {
            if(i==this->observation_particle_size)
            {
                os << "||||";
            }
            os << " " << i << " = " << *((this->particles_)[i]) << endl;
        }
		
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
			<< ", u:" << qnode->upper_bound();
                if(Globals::config.estimate_upper_bound)
                {
                    os << ", eu" << qnode->estimated_upper_bound();
                }
			os<< ", r:" << qnode->step_reward << ")" << endl;

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
	vstar(NULL),
        count_(0){
}

QNode::QNode(std::vector<State*>& particles):
particles_(particles), parent_(NULL),count_(0)
{
    
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
        lower_bound_alpha_vector.clear();
        upper_bound_alpha_vector.clear();
        default_upper_bound_alpha_vector.clear();
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
 double QNode::estimated_upper_bound() const {
        if(has_estimated_upper_bound_value)
        {
            return estimated_upper_bound_;
        }
        else
        {
            return upper_bound_;
        }
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

} // namespace despot
