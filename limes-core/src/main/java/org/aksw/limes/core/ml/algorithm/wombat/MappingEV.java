package org.aksw.limes.core.ml.algorithm.wombat;

import java.util.List;

public class MappingEV {
	private Integer num;
	private String sourceUri;
	private String targetUri;
	private double similarity;
	public MappingEV(Integer num, String sourceUri, String targetUri, double similarity) {
		this.num = num;
		this.sourceUri = sourceUri;
		this.targetUri = targetUri;
		this.similarity = similarity;
	}
	public String getSourceUri() {
		return sourceUri;
	}
	public void setSourceUri(String sourceUri) {
		this.sourceUri = sourceUri;
	}
	public String getTargetUri() {
		return targetUri;
	}
	public void setTargetUri(String targetUri) {
		this.targetUri = targetUri;
	}
	public double getSimilarity() {
		return similarity;
	}
	public void setSimilarity(double similarity) {
		this.similarity = similarity;
	}
	public Integer getNum() {
		return num;
	}
	public void setNum(Integer num) {
		this.num = num;
	}

}
