package org.aksw.limes.core.io.mapping;

import java.util.HashMap;

/**
 * @author Mohamed Sherif {@literal <}sherif {@literal @} informatik.uni-leipzig.de{@literal >}
 * @version Nov 12, 2015
 */
public class FileMapping extends AMapping {

	/**
	 * 
	 */
	private static final long serialVersionUID = -6896787320093743557L;
	public HashMap<String, HashMap<String, Double>> map;

    public int getNumberofMappings() {
        // TODO Auto-generated method stub
        return 0;
    }

    public FileMapping reverseSourceTarget() {
        // TODO Auto-generated method stub
        return null;
    }

    public int size() {
        // TODO Auto-generated method stub
        return 0;
    }

    public void add(String key, String value, Double double1) {
        // TODO Auto-generated method stub

    }

    public void add(String key, HashMap<String, Double> hashMap) {
        // TODO Auto-generated method stub

    }

    public double getConfidence(String key, String value) {
        // TODO Auto-generated method stub
        return 0.0d;
    }

    @Override
    public void add(String key, String value, double sim) {
        // TODO Auto-generated method stub

    }

    @Override
    public boolean contains(String key, String value) {
        // TODO Auto-generated method stub
        return false;
    }

    @Override
    public AMapping getBestOneToNMapping() {
        // TODO Auto-generated method stub
        return null;
    }

    @Override
    public AMapping getSubMap(double threshold) {
        // TODO Auto-generated method stub
        return null;
    }

	@Override
	public int getNumberofPositiveMappings() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public AMapping getOnlyPositiveExamples() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public AMapping getSubMap(double d, double e) {
		// TODO Auto-generated method stub
		return null;
	}

}
