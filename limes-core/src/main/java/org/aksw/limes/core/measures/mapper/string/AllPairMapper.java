package org.aksw.limes.core.measures.mapper.string;

import java.util.ArrayList;
import java.util.HashMap;

import org.aksw.limes.core.io.cache.ACache;
import org.aksw.limes.core.io.mapping.AMapping;
import org.aksw.limes.core.measures.mapper.AMapper;

public class AllPairMapper extends AMapper{

	double res_num;
	double cand_num;
	int THRESHOLD;
	int Q;

	ArrayList<String> records = new ArrayList<String>();
	ArrayList<ArrayList<Integer>> tokens = new ArrayList<ArrayList<Integer>>();

	int widow_bound = -1;

	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

	int overlap(int x, int y) {
		int result = 0;
		int posx = 0;
		int posy = 0;
		while (posx < tokens.get(x).size() && posy < tokens.get(y).size()) {
			if (tokens.get(x).get(posx) == tokens.get(y).get(posy))

			{
				result++;
				posx++;
				posy++;
			} else if (tokens.get(x).get(posx) < tokens.get(y).get(posy)) {
				posx++;
			} else {
				posy++;
			}
		}
		return result;
	}

	public static int DJBHash(String str) {
		long hash = 5381;

		for (int k = 0; k < str.length(); k++) {
			hash += (hash << 5) + str.charAt(k);
		}

		for (int k = 0; k < str.length(); k++) {
			hash += (hash << 5) + '$';
		}

		return (int) (hash & 0x7FFFFFFF);
	}

	void get_grams(int range_bound)
	{
        HashMap<Integer, Integer> freq_map = new HashMap<>(); 

		for (int k = range_bound; k < (int)records.size(); k++)
		{
			for (int sp = 0; sp < (int)records.get(k).length() - Q + 1; sp++)
			{
				int token = DJBHash(records.get(k));
				tokens.get(k).add(token);
			}
		}
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

	@Override
	public AMapping getMapping(ACache source, ACache target, String sourceVar, String targetVar, String expression,
			double threshold) {
		// TODO Auto-generated method stub
		return null;
	}

}
