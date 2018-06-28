#ifndef DESPOT_H
#define DESPOT_H

#include <despot/core/solver.h>
#include <despot/core/pomdp.h>
#include <despot/core/belief.h>
#include <despot/core/node.h>
#include <despot/core/globals.h>
#include <despot/core/history.h>
#include <despot/random_streams.h>

namespace despot {

//Class with functions that can be overriden by solver calculating lower bound using deep policy
class DespotStaticFunctionOverrideHelper {
    
public:
    DespotStaticFunctionOverrideHelper(){
    //name = "despot";
    }
    virtual ~DespotStaticFunctionOverrideHelper(){}
    virtual void InitMultipleLowerBounds(VNode* vnode, ScenarioLowerBound* lower_bound,
		RandomStreams& streams, History& history, 
                ScenarioLowerBound* learned_lower_bound = NULL, SearchStatistics* statistics = NULL){
       // std::cout << "Initializing multiple bounds from despot " << std::endl;
    }
    //Function returns time used by learned policy to get lower bound
    virtual double GetTimeNotToBeCounted(SearchStatistics* statistics) {
        //std::cout << "Calling from despot" << std::endl;
        return 0.0;
    }
    virtual void UpdateHistoryDuringTrial(History& history,VNode* vnode )
    {
         history.Add(vnode->parent()->edge(), vnode->edge());
    }
    
    virtual void Expand(QNode* qnode, ScenarioLowerBound* lower_bound,
		ScenarioUpperBound* upper_bound, const DSPOMDP* model,
		RandomStreams& streams, History& history,
                ScenarioLowerBound* learned_lower_bound = NULL, 
                SearchStatistics* statistics = NULL,
                DespotStaticFunctionOverrideHelper* o_helper=NULL);
    
    virtual void Update(QNode* qnode);
    virtual int GetObservationParticleSize(VNode* vnode)
    {
        return -1;
    }

    Solver *solver_pointer;
    //std::string name;
};    
class DESPOT: public Solver {
friend class VNode;

protected:
	VNode* root_;
	SearchStatistics* statistics_;

	ScenarioLowerBound* lower_bound_;
	ScenarioUpperBound* upper_bound_;
        DespotStaticFunctionOverrideHelper* o_helper_;

public:
	DESPOT(const DSPOMDP* model, ScenarioLowerBound* lb, ScenarioUpperBound* ub, Belief* belief = NULL);
	virtual ~DESPOT();

	ValuedAction Search();

	void belief(Belief* b);
	void Update(int action, OBS_TYPE obs);

	ScenarioLowerBound* lower_bound() const;
	ScenarioUpperBound* upper_bound() const;

	static VNode* ConstructTree(std::vector<State*>& particles, RandomStreams& streams,
		ScenarioLowerBound* lower_bound, ScenarioUpperBound* upper_bound,
		const DSPOMDP* model, History& history, double timeout,
		SearchStatistics* statistics = NULL, 
                ScenarioLowerBound* learned_lower_bound = NULL,
                DespotStaticFunctionOverrideHelper* o_helper=NULL);

public:
	static VNode* Trial(VNode* root, RandomStreams& streams,
		ScenarioLowerBound* lower_bound, ScenarioUpperBound* upper_bound,
		const DSPOMDP* model, History& history, SearchStatistics* statistics =
			NULL, ScenarioLowerBound* learned_lower_bound = NULL, 
                DespotStaticFunctionOverrideHelper* o_helper=NULL);
        
	static void InitLowerBound(VNode* vnode, ScenarioLowerBound* lower_bound,
		RandomStreams& streams, History& history, 
                ScenarioLowerBound* learned_lower_bound = NULL, 
                SearchStatistics* statistics = NULL, 
                DespotStaticFunctionOverrideHelper* o_helper=NULL);
	static void InitUpperBound(VNode* vnode, ScenarioUpperBound* upper_bound,
		RandomStreams& streams, History& history,
                DespotStaticFunctionOverrideHelper* o_helper=NULL);
	static void InitBounds(VNode* vnode, ScenarioLowerBound* lower_bound,
		ScenarioUpperBound* upper_bound, RandomStreams& streams, History& history, 
                ScenarioLowerBound* learned_lower_bound = NULL, 
                SearchStatistics* statistics = NULL, 
                DespotStaticFunctionOverrideHelper* o_helper=NULL);

	static void Expand(VNode* vnode,
		ScenarioLowerBound* lower_bound, ScenarioUpperBound* upper_bound,
		const DSPOMDP* model, RandomStreams& streams, History& history,
                ScenarioLowerBound* learned_lower_bound = NULL, 
                SearchStatistics* statistics = NULL,
                DespotStaticFunctionOverrideHelper* o_helper=NULL);
	static void Backup(VNode* vnode, DespotStaticFunctionOverrideHelper* o_helper=NULL);

	static double Gap(VNode* vnode);

	double CheckDESPOT(const VNode* vnode, double regularized_value);
	double CheckDESPOTSTAR(const VNode* vnode, double regularized_value);
	void Compare();

	static void ExploitBlockers(VNode* vnode);
	static VNode* FindBlocker(VNode* vnode);
	static void Expand(QNode* qnode, ScenarioLowerBound* lower_bound,
		ScenarioUpperBound* upper_bound, const DSPOMDP* model,
		RandomStreams& streams, History& history,
                ScenarioLowerBound* learned_lower_bound = NULL, 
                SearchStatistics* statistics = NULL, 
                DespotStaticFunctionOverrideHelper* o_helper=NULL);
	static void Update(VNode* vnode);
	static void Update(QNode* qnode);
	static VNode* Prune(VNode* vnode, int& pruned_action, double& pruned_value);
	static QNode* Prune(QNode* qnode, double& pruned_value);
	static double WEU(VNode* vnode);
	static double WEU(VNode* vnode, double epsilon);
	static VNode* SelectBestWEUNode(QNode* qnode);
	static QNode* SelectBestUpperBoundNode(VNode* vnode);
	static ValuedAction OptimalAction(VNode* vnode);

	static ValuedAction Evaluate(VNode* root, std::vector<State*>& particles,
		RandomStreams& streams, POMCPPrior* prior, const DSPOMDP* model);
        virtual void InitStatistics();
        virtual void CoreSearch(std::vector<State*>& particles, RandomStreams& streams );
        static int NumActions;
};

} // namespace despot

#endif
