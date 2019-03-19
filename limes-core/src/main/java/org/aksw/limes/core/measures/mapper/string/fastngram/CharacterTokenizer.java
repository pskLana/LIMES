package org.aksw.limes.core.measures.mapper.string.fastngram;

import java.util.HashSet;
import java.util.Set;

public class CharacterTokenizer implements ITokenizer {

	@Override
	public Set<String> tokenize(String s, int q) {
		if (s == null) {
			s = "";
		}
		Set<String> tokens = new HashSet<>();
		String[] chars = s.split("");
		for (String str : chars) {
			tokens.add(str);
		}
		return tokens;
	}

}
