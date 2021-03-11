package org.aksw.limes.core.ml.algorithm.wombat;

import java.util.ArrayList;
import java.util.List;

import org.aksw.limes.core.io.mapping.AMapping;

public class FullMappingEV implements Cloneable {

	private List<MappingEV> mEV;
	private List<List<Double>> stateEV;
	
	@Override
	public Object clone() throws CloneNotSupportedException {
        FullMappingEV cloned = (FullMappingEV) super.clone();
        cloned.mEV = new ArrayList<>();
        for (MappingEV item : mEV) {
        	cloned.mEV.add((MappingEV) item.clone());
        }
        cloned.stateEV = new ArrayList<>();
        for (List<Double> item : stateEV) {
        	cloned.stateEV.add(new ArrayList<>(item));
        }
        return cloned;
    }

	public FullMappingEV(List<MappingEV> mEV, List<List<Double>> stateEV) {
		this.mEV = mEV;
		this.stateEV = stateEV;
	}

	public List<MappingEV> getMappingEV() {
		return mEV;
	}
	
	public MappingEV getMappingByNum(Integer num) { 
		for(MappingEV i : mEV) {
			if(i.getNum().equals(num)) {
				return i;
			}
		}
//		return mEV.get(num);
		return null;
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

	public FullMappingEV remove(List<Integer> exampleNums) {
		for(Integer num: exampleNums){
        	MappingEV example = this.getMappingByNum(num);
            mEV.remove(example);
            stateEV.remove(stateEV.get(num));
        }
		return this;
	}

	public FullMappingEV join(FullMappingEV nextState) {
//		List<String> newList = new ArrayList<String>(listOne);
		this.mEV.addAll(nextState.getMappingEV());
		this.stateEV.addAll(nextState.getStateEV());
		return this;
	}

}
