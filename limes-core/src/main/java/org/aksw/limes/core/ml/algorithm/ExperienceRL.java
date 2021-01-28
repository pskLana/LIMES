package org.aksw.limes.core.ml.algorithm;

import java.util.List;

import org.aksw.limes.core.io.mapping.AMapping;
import org.aksw.limes.core.ml.algorithm.wombat.FullMappingEV;

public class ExperienceRL {
	FullMappingEV state;
	List<Integer> actions; 
	FullMappingEV nextState;
	double reward;
	public ExperienceRL(FullMappingEV m, List<Integer> exampleNums, FullMappingEV nextState, double reward) {
		this.state = m;
		this.actions = exampleNums;
		this.nextState = nextState;
		this.reward = reward;
	}
	public FullMappingEV getState() {
		return state;
	}
	public void setState(FullMappingEV state) {
		this.state = state;
	}
	public List<Integer> getActions() {
		return actions;
	}
	public void setActions(List<Integer> actions) {
		this.actions = actions;
	}
	public FullMappingEV getNextState() {
		return nextState;
	}
	public void setNextState(FullMappingEV nextState) {
		this.nextState = nextState;
	}
	public double getReward() {
		return reward;
	}
	public void setReward(double reward) {
		this.reward = reward;
	}

}
