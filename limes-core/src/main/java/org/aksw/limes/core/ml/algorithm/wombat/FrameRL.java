package org.aksw.limes.core.ml.algorithm.wombat;

public class FrameRL {
	public String source;
	public String target;
	public double similarity;
	public String sProp;
	public String tProp;
	public int result;
	public FrameRL(String source, String target,
			double similarity, String sProp, String tProp, int result) {
		this.source = source;
		this.target = target;
		this.similarity = similarity;
		this.sProp = sProp;
		this.tProp = tProp;
		this.result = result;
	}
	public String getSource() {
		return source;
	}
	public void setSource(String source) {
		this.source = source;
	}
	public String getTarget() {
		return target;
	}
	public void setTarget(String target) {
		this.target = target;
	}
	public double getSimilarity() {
		return similarity;
	}
	public void setSimilarity(double similarity) {
		this.similarity = similarity;
	}
	public String getsProp() {
		return sProp;
	}
	public void setsProp(String sProp) {
		this.sProp = sProp;
	}
	public String gettProp() {
		return tProp;
	}
	public void settProp(String tProp) {
		this.tProp = tProp;
	}
	public int getResult() {
		return result;
	}
	public void setResult(int result) {
		this.result = result;
	}

}
