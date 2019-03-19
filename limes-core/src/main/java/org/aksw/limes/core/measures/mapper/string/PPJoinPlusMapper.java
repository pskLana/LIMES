package org.aksw.limes.core.measures.mapper.string;

import org.aksw.limes.core.io.cache.ACache;
import org.aksw.limes.core.io.mapping.AMapping;
import org.aksw.limes.core.measures.mapper.AMapper;

public class PPJoinPlusMapper extends AMapper {
	
	/*Ppjoin combines positional filtering with prefix filtering


	y = [ A, B, C, D, E ]
	x = [ B, C, D, E, F ]

	since they share a common token, B, in their prefixes,
	prefix filtering-based methods will select y as a candidate


	we can obtain an estimate of the maximum possible overlap as the sum of
	current overlap amount and the minimum number of unseen tokens in x and y, that
	is, 1 + min(3, 4) = 4. 

	Each element in the posting list is of the form postings list of w (i.e., Iw) is of the form (rid, pos), indicating that the pos-th token in
	record rid’s prefix is w

	Ppjoin plus uses suffix filtering in addition
*/
	@Override
	public AMapping getMapping(ACache source, ACache target, String sourceVar, String targetVar, String expression,
			double threshold) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public double getRuntimeApproximation(int sourceSize, int targetSize, double theta, Language language) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double getMappingSizeApproximation(int sourceSize, int targetSize, double theta, Language language) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public String getName() {
		// TODO Auto-generated method stub
		return null;
	}

}
