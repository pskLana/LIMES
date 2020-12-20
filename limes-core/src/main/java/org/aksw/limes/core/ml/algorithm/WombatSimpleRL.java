package org.aksw.limes.core.ml.algorithm;

import java.util.*;
import java.util.Map.Entry;

import org.aksw.limes.core.datastrutures.GoldStandard;
import org.aksw.limes.core.datastrutures.LogicOperator;
import org.aksw.limes.core.datastrutures.Tree;
import org.aksw.limes.core.evaluation.qualititativeMeasures.FMeasure;
import org.aksw.limes.core.evaluation.qualititativeMeasures.PseudoFMeasure;
import org.aksw.limes.core.exceptions.UnsupportedMLImplementationException;
import org.aksw.limes.core.execution.planning.plan.Instruction;
import org.aksw.limes.core.execution.planning.plan.Instruction.Command;
import org.aksw.limes.core.io.cache.ACache;
import org.aksw.limes.core.io.ls.LinkSpecification;
import org.aksw.limes.core.io.mapping.AMapping;
import org.aksw.limes.core.io.mapping.MappingFactory;
import org.aksw.limes.core.io.mapping.MappingFactory.MappingType;
import org.aksw.limes.core.io.mapping.reader.AMappingReader;
import org.aksw.limes.core.io.mapping.reader.CSVMappingReader;
import org.aksw.limes.core.measures.mapper.MappingOperations;
import org.aksw.limes.core.measures.measure.AMeasure;
import org.aksw.limes.core.measures.measure.MeasureFactory;
import org.aksw.limes.core.measures.measure.MeasureType;
import org.aksw.limes.core.measures.measure.string.JaccardMeasure;
import org.aksw.limes.core.ml.algorithm.classifier.ExtendedClassifier;
import org.aksw.limes.core.ml.algorithm.wombat.AWombat;
import org.aksw.limes.core.ml.algorithm.wombat.EnvRL;
import org.aksw.limes.core.ml.algorithm.wombat.FrameRL;
import org.aksw.limes.core.ml.algorithm.wombat.FullMappingEV;
import org.aksw.limes.core.ml.algorithm.wombat.LinkEntropy;
import org.aksw.limes.core.ml.algorithm.wombat.MappingEV;
import org.aksw.limes.core.ml.algorithm.wombat.RefinementNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import jep.Interpreter;
import jep.JepException;
import jep.SharedInterpreter;

/**
 * Simple implementation of the Wombat algorithm
 * Fast implementation, that is not complete
 *
 * @author Mohamed Sherif (sherif@informatik.uni-leipzig.de)
 * @version Jun 7, 2016
 */
public class WombatSimpleRL extends AWombat {

    private static final Logger logger = LoggerFactory.getLogger(WombatSimpleRL.class);

    private static final String ALGORITHM_NAME = "Wombat Simple RL";

    private static final int activeLearningRate = 3;

    private RefinementNode bestSolutionNode = null;

    private List<ExtendedClassifier> classifiers = null;

    private Tree<RefinementNode> refinementTreeRoot = null;
    private ACache sourceInstance = null;
    private ACache targetInstance = null;
    Object d = null;
    Interpreter interp = null;
    double oldFMeasure = 0.0;
    
    private boolean firstIter = false;
    private AMapping groundTruthExamples = null;
    /**
     * WombatSimple constructor.
     */
    protected WombatSimpleRL() {
        super();
    }

    @Override
    protected String getName() {
        return ALGORITHM_NAME;
    }

    @Override
    protected void init(List<LearningParameter> lp, ACache sourceCache, ACache targetCache) {
        super.init(lp, sourceCache, targetCache);
        sourceUris = sourceCache.getAllUris();
        targetUris = targetCache.getAllUris();
        sourceInstance = sourceCache;
        targetInstance = targetCache;
        bestSolutionNode = null;
        classifiers = null;
        try {
			interp = new SharedInterpreter();
			interp.runScript("src/main/resources/DQN.py");
		} catch (JepException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }

    @Override
    protected MLResults learn(AMapping trainingData) {
        this.trainingData = trainingData;
        fillSampleSourceTargetCaches(trainingData);
        this.sourceCache = sourceSample;
        this.targetCache = targetSample;
        return learn();
    }

    /**
     * @return wrap with results, null if no result found
     */
    private MLResults learn() {
        if (bestSolutionNode == null) { // not to do learning twice
            bestSolutionNode = findBestSolution();
        }
        String bestMetricExpr = bestSolutionNode.getMetricExpression();
        if(!bestMetricExpr.equals("")){
            double threshold = Double.parseDouble(bestMetricExpr.substring(bestMetricExpr.lastIndexOf("|") + 1, bestMetricExpr.length()));
            AMapping bestMapping = bestSolutionNode.getMapping();
            LinkSpecification bestLS = new LinkSpecification(bestMetricExpr, threshold);
            double bestFMeasure = bestSolutionNode.getFMeasure();
            return new MLResults(bestLS, bestMapping, bestFMeasure, null);
        }
        // case no mapping found
        return null;
    }

    @Override
    protected MLResults learn(PseudoFMeasure pfm) {
        if(pfm != null){
            this.pseudoFMeasure = pfm;
        }else{ // use default PFM
            this.pseudoFMeasure = new PseudoFMeasure();
        }
        this.isUnsupervised = true;
        return learn();
    }

    @Override
    protected boolean supports(MLImplementationType mlType) {
        return mlType == MLImplementationType.SUPERVISED_BATCH ||
                mlType == MLImplementationType.UNSUPERVISED    ||
                mlType == MLImplementationType.SUPERVISED_ACTIVE;
    }

    protected AMapping getNextExamplesWombatSimple(int size) throws UnsupportedMLImplementationException {
        List<RefinementNode> bestNodes = getBestKNodes(refinementTreeRoot, activeLearningRate);
        AMapping intersectionMapping = MappingFactory.createDefaultMapping();
        AMapping unionMapping = MappingFactory.createDefaultMapping();

        for(RefinementNode sn : bestNodes){
            intersectionMapping = MappingOperations.intersection(intersectionMapping, sn.getMapping());
            unionMapping = MappingOperations.union(unionMapping, sn.getMapping());
        }
        AMapping posEntropyMapping = MappingOperations.difference(unionMapping, intersectionMapping);

        TreeSet<LinkEntropy> linkEntropy = new TreeSet<>();
        int entropyPos = 0, entropyNeg = 0;
        for(String s : posEntropyMapping.getMap().keySet()){
            for(String t : posEntropyMapping.getMap().get(s).keySet()){
                // compute Entropy(s,t)
                for(RefinementNode sn : bestNodes){
                    if(sn.getMapping().contains(s, t)){
                        entropyPos++;
                    }else{
                        entropyNeg++;
                    }
                }
                int entropy = entropyPos * entropyNeg;
                linkEntropy.add(new LinkEntropy(s, t, entropy));
            }
        }
        // get highestEntropyLinks
        List<LinkEntropy> highestEntropyLinks = new ArrayList<>();
        int i = 0;
        Iterator<LinkEntropy> itr = linkEntropy.descendingIterator();
        while(itr.hasNext() && i < size) {
            LinkEntropy next = itr.next();
            if (!trainingData.contains(next.getSourceUri(), next.getTargetUri())) {
                highestEntropyLinks.add(next);
                i++;
            }
        }
        AMapping result = MappingFactory.createDefaultMapping();
        for(LinkEntropy l: highestEntropyLinks){
            result.add(l.getSourceUri(), l.getTargetUri(), l.getEntropy());
        }
        
        return result;
    }

    @Override
    protected MLResults activeLearn(){
        return learn(new PseudoFMeasure());
    }

    @Override
    protected MLResults activeLearn(AMapping oracleMapping) {
        trainingData = MappingOperations.union(trainingData, oracleMapping);
        boolean hasPositiveExamples = trainingData.getMap().entrySet().stream()
                .flatMap(e -> e.getValue().values().stream())
                .anyMatch(x -> x == 1);
        if (hasPositiveExamples) {
            updateScores(refinementTreeRoot);
            isUnsupervised = false;
            bestSolutionNode = findBestSolution();
        }
        return learn();
    }


    /**
     * update F-Measure of the refinement tree r
     * based on either training data or PFM
     *
     * @param r refinement tree
     */
    protected void updateScores(Tree<RefinementNode> r) {
        if (r.getchildren() == null || r.getchildren().size() == 0) {
            r.getValue().setfMeasure(fMeasure(r.getValue().getMapping()));
            return;
        }
        for (Tree<RefinementNode> child : r.getchildren()) {
            if (child.getValue().getFMeasure() >= 0) {
                r.getValue().setfMeasure(fMeasure(r.getValue().getMapping()));
                updateScores(child);
            }
        }
    }


    /**
     * @return RefinementNode containing the best over all solution
     */
    public RefinementNode findBestSolution() {
        classifiers = findInitialClassifiers();
        createRefinementTreeRoot();
        Tree<RefinementNode> mostPromisingNode = getMostPromisingNode(refinementTreeRoot);
        logger.debug("Most promising node: " + mostPromisingNode.getValue());
        int i = 1;
        while ((mostPromisingNode.getValue().getFMeasure()) < getMaxFitnessThreshold()
                && refinementTreeRoot.size() <= getMaxRefinmentTreeSize()
                && i <= getMaxIterationNumber()) {
            expandNode(mostPromisingNode);
            mostPromisingNode = getMostPromisingNode(refinementTreeRoot);
            if (mostPromisingNode.getValue().getFMeasure() == -Double.MAX_VALUE) {
                break; // no better solution can be found
            }
            logger.debug("Most promising node: " + mostPromisingNode.getValue());
            i++;
        }
        RefinementNode bestSolution = getBestNode(refinementTreeRoot).getValue();
        logger.debug("Overall Best Solution: " + bestSolution);
        return bestSolution;
    }



    /**
     * @param r the root of the refinement tree
     * @param k number of best nodes
     * @return sorted list of best k tree nodes
     */
    protected List<RefinementNode> getBestKNodes(Tree<RefinementNode> r, int k) {
        TreeSet<RefinementNode> ts = new TreeSet<>();
        TreeSet<RefinementNode> sortedNodes = getSortedNodes(r, getOverAllPenaltyWeight(), ts);
        List<RefinementNode> resultList = new ArrayList<>();
        int i = 0;
        Iterator<RefinementNode> itr = sortedNodes.descendingIterator();
        while(itr.hasNext() && i < k) {
            resultList.add(itr.next());
            i++;
        }
        return resultList;
    }


    /**
     * @param r the root of the refinement tree
     * @param penaltyWeight from 0 to 1
     * @param result refinment tree
     * @return sorted list of tree nodes
     */
    protected TreeSet<RefinementNode> getSortedNodes(Tree<RefinementNode> r, double penaltyWeight, TreeSet<RefinementNode> result) {
        if (r.getchildren() == null || r.getchildren().size() == 0) {
            result.add(r.getValue());
            return result;
        }
        for (Tree<RefinementNode> child : r.getchildren()) {
            if (child.getValue().getFMeasure() >= 0) {
                result.add(r.getValue());
                return getSortedNodes(child, penaltyWeight, result);
            }
        }
        return null;
    }

    /**
     * Expand an input refinement node by applying
     * all available operators to the input refinement
     * node's mapping with all other classifiers' mappings
     *
     * @param node
     *         Refinement node to be expanded
     * @author sherif
     */
    private void expandNode(Tree<RefinementNode> node) {
        AMapping map = MappingFactory.createDefaultMapping();
        for (ExtendedClassifier c : classifiers) {
            for (LogicOperator op : LogicOperator.values()) {
                if (!node.getValue().getMetricExpression().equals(c.getMetricExpression())) { // do not create the same metricExpression again
                    if (op.equals(LogicOperator.AND)) {
                        map = MappingOperations.intersection(node.getValue().getMapping(), c.getMapping());
                    } else if (op.equals(LogicOperator.OR)) {
                        map = MappingOperations.union(node.getValue().getMapping(), c.getMapping());
                    } else if (op.equals(LogicOperator.MINUS)) {
                        map = MappingOperations.difference(node.getValue().getMapping(), c.getMapping());
                    }
                    String metricExpr = op + "(" + node.getValue().getMetricExpression() + "," + c.getMetricExpression() + ")|0";
                    RefinementNode child = createNode(map, metricExpr);
                    node.addChild(new Tree<RefinementNode>(child));
                }
            }
        }
        if (isVerbose()) {
            refinementTreeRoot.print();
        }
    }



    /**
     * initiate the refinement tree as a root node  with set of
     * children nodes containing all initial classifiers
     *
     */
    protected void createRefinementTreeRoot() {
        RefinementNode initialNode = new RefinementNode(-Double.MAX_VALUE, MappingFactory.createMapping(MappingType.DEFAULT), "");
        refinementTreeRoot = new Tree<RefinementNode>(null, initialNode, null);
        for (ExtendedClassifier c : classifiers) {
            RefinementNode n = new RefinementNode(c.getfMeasure(), c.getMapping(), c.getMetricExpression());
            refinementTreeRoot.addChild(new Tree<RefinementNode>(refinementTreeRoot, n, null));
        }
        if (isVerbose()) {
            refinementTreeRoot.print();
        }
    }

	@Override
	protected AMapping getNextExamples(int size) throws UnsupportedMLImplementationException {
		if(!firstIter) {
//			groundTruthExamples = getNextExamplesWombatSimple(size);
			int N = 10; // for the decision boundary
			AMapping state = getLSandState(N);
			// save state as embedding vectors in stateEV
			FullMappingEV m = getStateAsEV(state);// contains mapping for later and EVs
//			for (List<Double> temp : stateEV) {
//	            System.out.println(temp.size());
//	        }
			List<Integer> exampleNums = new ArrayList<Integer>();
			// train NN and get examples to show to the user
			exampleNums = trainNNandGetExamples(m.getStateEV());
			
			AMapping result = MappingFactory.createDefaultMapping();
	        for(Integer num: exampleNums){

	        	MappingEV example = m.getMappingByNum(num);
	            result.add(example.getSourceUri(), example.getTargetUri(), example.getSimilarity());  	
	        }
	        
			firstIter = true;
			
			return result;
		} else {
			runNextEpisode(); 
		}
		return null;
	}
	
	public FullMappingEV getStateAsEV(AMapping state) {
		// get embedding vectors for person11
		String dataPerson11Path = "src/main/resources/ConEx-vectors/Person1/person11.csv";
		AMappingReader mappingReader;
    	mappingReader = new CSVMappingReader(dataPerson11Path);
    	HashMap<String, List<Double>> person11DataMap = new HashMap<String, List<Double>>();
        person11DataMap = ((CSVMappingReader) mappingReader).readEV();
        
        // get embedding vectors for person12
        String dataPerson12Path = "src/main/resources/ConEx-vectors/Person1/person12.csv";
		AMappingReader mappingReader1;
    	mappingReader1 = new CSVMappingReader(dataPerson12Path);
    	HashMap<String, List<Double>> person12DataMap = new HashMap<String, List<Double>>();
        person12DataMap = ((CSVMappingReader) mappingReader1).readEV();
        
        List<List<Double>> stateEV = new ArrayList<List<Double>>();
        Integer num = 0;
        List<MappingEV> mEV = new ArrayList<MappingEV>();
        for (Entry<String, HashMap<String, Double>> entry : state.getMap().entrySet()) {
            for (Entry<String, Double> items : entry.getValue().entrySet()) {
            	List<Double> newList = new ArrayList<Double>();
            	newList.addAll(person11DataMap.get(entry.getKey()));
            	newList.addAll(person12DataMap.get(items.getKey()));
            	stateEV.add(newList);
            	MappingEV mappingEV = new MappingEV(num, entry.getKey(), items.getKey(), items.getValue());
            	mEV.add(mappingEV);
            	num++;
            }
        }
        FullMappingEV m = new FullMappingEV(mEV, stateEV);
        return m;
	}
	
	public List<Integer> trainNNandGetExamples(List<List<Double>> stateEV) {
		// run python script with DQL 	        
        try {
            
            // any of the following work, these are just pseudo-examples
            
            // using exec(String) to invoke methods
	        interp.set("arg", stateEV);
	        interp.set("WombatRLObject", this);
	        interp.exec("x = mainFun(arg)");
//            interp.exec("x = mainFun()");
            Object action = interp.getValue("x"); // so far we take only K=1
//            d = action;
            Long d = (long) action;
            System.out.println(action);
            List<Integer> exampleNums = new ArrayList<Integer>();
            exampleNums.add(d.intValue());
            return exampleNums;

        }
        catch (JepException e) {
        	// TODO Auto-generated catch block
        	e.printStackTrace();
        }
        
		return null;
	}
	
	public List<FrameRL> runNextWombatAndGetRandomExamples(int k) {
		MLResults mlm = this.learn(this.trainingData);
        logger.info("Learned: " + mlm.getLinkSpecification().getFullExpression() + " with threshold: " + mlm.getLinkSpecification().getThreshold());
        // Applying 10 random examples from source and target to link spec
        List<String> randomSourceList = givenList_whenNumberElementsChosen_shouldReturnRandomElementsNoRepeat(sourceUris);
        List<String> randomTargetList = givenList_whenNumberElementsChosen_shouldReturnRandomElementsNoRepeat(targetUris);
        System.out.println(mlm.getLinkSpecification().getMeasure());//setThreshold(0.5);
        String str = "jaccard(x.pref0:given_name,y.pref1:given_name)";// mlm.getLinkSpecification().getMeasure()
        Instruction inst = new Instruction(
        		Command.RUN, str, 
        		Double.toString(0.6), -1, -1, 0);
        MeasureType type = MeasureFactory.getMeasureType(inst.getMeasureExpression());
        AMeasure measure = MeasureFactory.createMeasure(type);
//        Map<String, Double> stMeasure  = new HashMap<String, Double>();
        List<FrameRL> stMeasure  = new ArrayList<FrameRL>();
        int amountOfPositive = 0;
        for (String s : randomSourceList) {
        	for (String t : randomTargetList) {
        		int positive = -1;
//        		String prop1 = mlm.getLinkSpecification().getProperty1(); How to get properties from link spec ("pref0:given_name") and ("pref1:given_name")???
        		TreeSet<String> sourceProp = sourceInstance.getInstance(s).getProperty("pref0:given_name");
        		TreeSet<String> targetProp = targetInstance.getInstance(t).getProperty("pref1:given_name");
        		double m = ((JaccardMeasure) measure).getSimilarityChar(sourceProp, targetProp);
//        		double m = measure.getSimilarity("b r a e n", "b r o e n");
//        		stMeasure.put("["+sourceProp+"]"+"["+targetProp+"]", m);
//        		stMeasure.put("["+s+"]"+"["+t+"]", m);
        		if(m >= 0.5) {
        			amountOfPositive++;
        			positive = 1;
        		}
        		FrameRL fr = new FrameRL(s, t, m, sourceProp.first(), targetProp.first(), positive);     		
        		stMeasure.add(fr);
            }
        }
        Collections.sort(stMeasure, Collections.reverseOrder());
        List<FrameRL> bestK = stMeasure.subList(0, k);
        return bestK;
	}
	
	public AMapping getLSandState(int N) {
		MLResults mlm = this.learn(this.trainingData);
        logger.info("Learned: " + mlm.getLinkSpecification().getFullExpression() + " with threshold: " + mlm.getLinkSpecification().getThreshold());
        String str = "jaccard(x.pref0:given_name,y.pref1:given_name)";// mlm.getLinkSpecification().getMeasure()
        Instruction inst = new Instruction(
        		Command.RUN, str, 
        		Double.toString(0.6), -1, -1, 0);
        MeasureType type = MeasureFactory.getMeasureType(inst.getMeasureExpression());
        AMeasure measure = MeasureFactory.createMeasure(type);
        List<FrameRL> stMeasure  = new ArrayList<FrameRL>();
        for (String s : sourceUris) {
        	for (String t : targetUris) {
        		TreeSet<String> sourceProp = sourceInstance.getInstance(s).getProperty("pref0:given_name");
        		TreeSet<String> targetProp = targetInstance.getInstance(t).getProperty("pref1:given_name");
        		double m = ((JaccardMeasure) measure).getSimilarityChar(sourceProp, targetProp);
        		FrameRL fr = new FrameRL(s, t, m, sourceProp.first(), targetProp.first(), 0);     		
        		stMeasure.add(fr);
            }
        }
        Collections.sort(stMeasure, Collections.reverseOrder());
        AMapping state = getNearestToBoundary(stMeasure, N);
        return state;
	}

	public void runNextEpisode() {
		
		// Initialization
		// Using supervised batch
		// return k-pairs with highest sim measure
		int k = 10;
		int m = 3;
		List<FrameRL> stMeasure = runNextWombatAndGetRandomExamples(k);
		
		// Start RL
		List<FrameRL> stMeasure1 = runNextWombatAndGetRandomExamples(m);
		        
		        //Compute decision boundary --> m >= 0.5
		        
		        
		        // run python script with DQL 	        
		        try {
		            
		            // any of the following work, these are just pseudo-examples
		            
		            // using exec(String) to invoke methods
			        interp.set("arg", stMeasure);
			        interp.set("WombatRLObject", this);
			        interp.exec("x = mainFun(arg)");
//		            interp.exec("x = mainFun()");
		            Object result1 = interp.getValue("x");
		            d = result1;
		            System.out.println(result1);

		        }
		        catch (JepException e) {
		        	// TODO Auto-generated catch block
		        	e.printStackTrace();
		        }
		
	}

	public List<String> givenList_whenNumberElementsChosen_shouldReturnRandomElementsNoRepeat(List<String> list) {
	    Random rand = new Random();
	    List<String> receivedList = new ArrayList<String>();
	    List<String> givenList = list;
	 
	    int numberOfElements = 10;
	 
	    for (int i = 0; i < numberOfElements; i++) {
	        int randomIndex = rand.nextInt(givenList.size());
	        String randomElement = givenList.get(randomIndex);
	        receivedList.add(randomElement);
	        givenList.remove(randomIndex);
	    }
	    return receivedList;
	}
	
    //Select an action. Via exploration or exploitation
    // 2 possible actions(take 3 best or take 3 nearest to 0.5)
    // We take a random action, for example, take 3 best.
	public EnvRL get3Best(List<FrameRL> stMeasure) throws JepException {
        Collections.sort(stMeasure, Collections.reverseOrder());
        List<FrameRL> best3 = stMeasure.subList(0, 3);
        
        AMapping result = MappingFactory.createDefaultMapping();
        for(FrameRL l: best3){
            result.add(l.getSource(), l.getTarget(), l.getSimilarity());
        }
        
        // calculate F-measure
        double newFMeasure = new FMeasure().calculate(result, new GoldStandard(trainingData), getBeta());
        System.out.println(newFMeasure);
        // reward
        double reward = newFMeasure - oldFMeasure;
//        // update old FMeasure
//        oldFMeasure = newFMeasure;
        EnvRL envRL = new EnvRL(best3, newFMeasure, reward); 
        interp.set("EnvRLObject", envRL);
        return envRL;
	}
	
	public AMapping getNearestToBoundary(List<FrameRL> stMeasure, int N) {
		Map<Double, FrameRL> distanceBetweenMeasures = new HashMap<Double, FrameRL>();
		for(FrameRL l: stMeasure){
			distanceBetweenMeasures.put(Math.abs(l.getSimilarity()-0.5), l);
        }
		TreeMap<Double, FrameRL> sorted = new TreeMap<Double, FrameRL>(distanceBetweenMeasures);

        List<FrameRL> best = new ArrayList<FrameRL>();
        int num = 0;
        for(Map.Entry<Double, FrameRL> entry : sorted.entrySet()) {
        	if(num >=N+N) {
        		break;
        	}
	    	Double key = entry.getKey();
	    	FrameRL value = entry.getValue();
	    	best.add(value);
	    	num++;
        }
        
        AMapping result = MappingFactory.createDefaultMapping();
        for(FrameRL l: best){
            result.add(l.getSource(), l.getTarget(), l.getSimilarity());
        }
        
//        // calculate F-measure
//        double newFMeasure = new FMeasure().calculate(result, new GoldStandard(trainingData), getBeta());
//        // reward
//        double reward = newFMeasure - oldFMeasure;
//        EnvRL envRL = new EnvRL(best3, newFMeasure, reward); 
//        interp.set("EnvRLObject", envRL);
//        return envRL;
        return result;
	}
	
//	replace 3 predicted examples with 3 worst ones (call Wombat, generate random examples and replace 3 worst ones)
//	sort new examples by similarityMeasure and replace 3 first worse examples with self.selectedExamples
	 public List<FrameRL> replaceWorstExamples(List<FrameRL> selectedExamples) {
		 List<FrameRL> stMeasure = runNextWombatAndGetRandomExamples(10);
		 Collections.sort(stMeasure, Collections.reverseOrder());
		 List<FrameRL> newExamples = new ArrayList<FrameRL>(selectedExamples);
		 newExamples.addAll(stMeasure.subList(3, stMeasure.size()));
		 return newExamples;
	 }

}


