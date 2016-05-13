package org.aksw.limes.core.measures.mapper.temporal.allenAlgebra.mappers.complex;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.TreeMap;

import org.aksw.limes.core.io.cache.Cache;
import org.aksw.limes.core.io.mapping.Mapping;
import org.aksw.limes.core.io.mapping.MemoryMapping;
import org.aksw.limes.core.measures.mapper.IMapper.Language;
import org.aksw.limes.core.measures.mapper.temporal.allenAlgebra.AllenAlgebraMapper;
import org.aksw.limes.core.measures.mapper.temporal.allenAlgebra.mappers.atomic.BeginEnd;
import org.aksw.limes.core.measures.mapper.temporal.allenAlgebra.mappers.atomic.EndBegin;

import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.Map.Entry;

public class Meets extends AllenAlgebraMapper {

    public Meets() {
	// EB0
	this.getRequiredAtomicRelations().add(4);
    }

    @Override
    public String getName() {
	return "Meets";
    }

    @Override
    public Mapping getMapping(ArrayList<TreeMap<String, Set<String>>> maps) {

	Mapping m = new MemoryMapping();

	TreeMap<String, Set<String>> mapEB0 = maps.get(0);

	for (Map.Entry<String, Set<String>> entryEB0 : mapEB0.entrySet()) {

	    String instancEB0 = entryEB0.getKey();
	    Set<String> setEB0 = entryEB0.getValue();

	    for (String targetInstanceUri : setEB0) {
		m.add(instancEB0, targetInstanceUri, 1);
	    }
	}
	return m;
    }
    @Override
    public Mapping getMapping(Cache source, Cache target, String sourceVar, String targetVar, String expression,
	    double threshold) {
	ArrayList<TreeMap<String, Set<String>>> maps = new ArrayList<TreeMap<String, Set<String>>>();
	EndBegin eb = new EndBegin();
	// EB0
	maps.add(eb.getConcurrentEvents(source, target, expression));
	Mapping m = getMapping(maps);
	return m;
    }

    public double getRuntimeApproximation(int sourceSize, int targetSize, double theta, Language language) {
	return 1000d;
    }

    public double getMappingSizeApproximation(int sourceSize, int targetSize, double theta, Language language) {
	return 1000d;
    }
}
