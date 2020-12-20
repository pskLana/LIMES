package org.aksw.limes.core.ml.algorithm.wombat;

import java.util.List;

public class FullMappingEV {

	private List<MappingEV> mEV;
	private List<List<Double>> stateEV;

	public FullMappingEV(List<MappingEV> mEV, List<List<Double>> stateEV) {
		this.mEV = mEV;
		this.stateEV = stateEV;
	}

	public List<MappingEV> getMappingEV() {
		return mEV;
	}
	
	public MappingEV getMappingByNum(Integer num) { 
		return mEV.get(num);
	}

	public void setMappingEV(List<MappingEV> mappingEV) {
		this.mEV = mappingEV;
	}

	public List<List<Double>> getStateEV() {
		return stateEV;
	}

	public void setStateEV(List<List<Double>> stateEV) {
		this.stateEV = stateEV;
	}

}
