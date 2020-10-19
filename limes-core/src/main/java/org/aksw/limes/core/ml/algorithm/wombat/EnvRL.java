package org.aksw.limes.core.ml.algorithm.wombat;

import java.util.List;

public class EnvRL {

	private List<FrameRL> selectedExamples;
	private double newFMeasure;
	private double reward;

	public EnvRL(List<FrameRL> selectedExamples, double newFMeasure, double reward) {
		this.selectedExamples = selectedExamples;
		this.newFMeasure = newFMeasure;
		this.reward = reward;
	}
	
	public List<FrameRL> getSelectedExamples() {
		return selectedExamples;
	}

	public void setSelectedExamples(List<FrameRL> selectedExamples) {
		this.selectedExamples = selectedExamples;
	}

	public double getNewFMeasure() {
		return newFMeasure;
	}

	public void setNewFMeasure(double newFMeasure) {
		this.newFMeasure = newFMeasure;
	}

	public double getReward() {
		return reward;
	}

	public void setReward(double reward) {
		this.reward = reward;
	}

}
