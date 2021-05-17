package org.aksw.limes.core.ml.algorithm;

import java.util.*;
import java.util.Map.Entry;

import org.aksw.limes.core.controller.LSPipeline;
import org.aksw.limes.core.datastrutures.GoldStandard;
import org.aksw.limes.core.datastrutures.LogicOperator;
import org.aksw.limes.core.datastrutures.Tree;
import org.aksw.limes.core.evaluation.qualititativeMeasures.FMeasure;
import org.aksw.limes.core.evaluation.qualititativeMeasures.PseudoFMeasure;
import org.aksw.limes.core.exceptions.UnsupportedMLImplementationException;
import org.aksw.limes.core.execution.engine.ExecutionEngine;
import org.aksw.limes.core.execution.engine.ExecutionEngineFactory;
import org.aksw.limes.core.execution.engine.ExecutionEngineFactory.ExecutionEngineType;
import org.aksw.limes.core.execution.planning.planner.ExecutionPlannerFactory;
import org.aksw.limes.core.execution.planning.planner.IPlanner;
import org.aksw.limes.core.execution.planning.planner.ExecutionPlannerFactory.ExecutionPlannerType;
import org.aksw.limes.core.execution.rewriter.Rewriter;
import org.aksw.limes.core.execution.rewriter.RewriterFactory;
import org.aksw.limes.core.io.cache.ACache;
import org.aksw.limes.core.io.ls.LinkSpecification;
import org.aksw.limes.core.io.mapping.AMapping;
import org.aksw.limes.core.io.mapping.MappingFactory;
import org.aksw.limes.core.io.mapping.MappingFactory.MappingType;
import org.aksw.limes.core.io.mapping.reader.AMappingReader;
import org.aksw.limes.core.io.mapping.reader.CSVMappingReader;
import org.aksw.limes.core.measures.mapper.MappingOperations;
import org.aksw.limes.core.ml.algorithm.classifier.ExtendedClassifier;
import org.aksw.limes.core.ml.algorithm.wombat.AWombat;
import org.aksw.limes.core.ml.algorithm.wombat.EnvRL;
import org.aksw.limes.core.ml.algorithm.wombat.FrameRL;
import org.aksw.limes.core.ml.algorithm.wombat.FullMappingEV;
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
    private List<ExperienceRL> experienceList = new ArrayList<ExperienceRL>();
    private double currentReward = 0.0;
//    public AMapping trainingData;
    private int counterAL = 0;
    private int experienceCounter = -1;
    private double decisionBoundaryTheshold = 0.8;
    private double epsilon = 0.3;//0.05;
    private int numberOfPairs = 40;
    private boolean firstStep = true;
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
        this.sourceCache = sourceCache;
        this.targetCache = targetCache;
        sourceSample = sourceCache;
        targetSample = targetCache;
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
			counterAL++;
//			groundTruthExamples = getNextExamplesWombatSimple(size);
			int N = 10; // for the decision boundary
			AMapping state = getLSandState(N);
			// save state as embedding vectors in stateEV
			FullMappingEV m = getStateAsEV(state, null);// contains mapping for later and EVs
//			for (List<Double> temp : stateEV) {
//	            System.out.println(temp.size());
//	        }
			List<Integer> exampleNums = new ArrayList<Integer>();
			
			// initializeRL in python
        	try {
				interp.exec("initializeRL()");
			} catch (JepException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
        	
			// train NN and get examples to show to the user
			exampleNums = trainNNandGetExamples(m.getStateEV(), false);
			
			AMapping result = getResultAndSaveExperience(m, exampleNums);
			
		    experienceCounter++;
		    firstIter = true;
	        			
			return result;
		} else {
//			runNextEpisode(); 
			
			// add K=1 new examples nearest to the decision boundary	
			AMapping newState = getLSandNextState(experienceList.get(experienceCounter).getActions().size()/2.0);
			if(newState == null) {
				System.out.println("Next state cannot be received");
				return null;
			}
			// save state as embedding vectors in stateEV
			FullMappingEV newM = getStateAsEV(newState, experienceList.get(experienceCounter).getActions());// contains mapping for later and EVs
			// join K new examples with old from the previous iteration 
			FullMappingEV m = newM.join(experienceList.get(experienceCounter).getNextState());
			experienceCounter++;
			List<Integer> exampleNums = new ArrayList<Integer>();
			// train NN and get examples to show to the user
			if (experienceCounter/counterAL == size) { // if this is the last iteration of AL
				exampleNums = trainNNandGetExamples(m.getStateEV(), true);
			} else {
				exampleNums = trainNNandGetExamples(m.getStateEV(), false);
			}
			
			AMapping result = getResultAndSaveExperience(m, exampleNums);
	        
	        logger.info("Current F-Measure: "+this.currentReward);
	        if (experienceCounter/counterAL == size) { // if this is the last iteration of AL
	        	firstIter = false; // preparation for the next episode of AL
	        	return MappingFactory.createDefaultMapping();
			} else {
				return result;
			}
	        
	        
		}
//		return null;
	}
	
	public AMapping getResultAndSaveExperience(FullMappingEV m, List<Integer> exampleNums) {
		AMapping result = MappingFactory.createDefaultMapping();
        for(Integer num: exampleNums){

        	MappingEV example = m.getMappingByNum(num);
            result.add(example.getSourceUri(), example.getTargetUri(), example.getSimilarity());  	
        }
        
        // save next state without chosen action and save experience
        // remove previous action from the state
        try {
			FullMappingEV m1 = (FullMappingEV) m.clone();
			FullMappingEV nextState = m1.remove(exampleNums); 
//			if(!firstIter) {
//				this.currentReward = 0.0;
//			}
	        experienceList.add(new ExperienceRL(m, exampleNums, nextState, this.currentReward));
		} catch (CloneNotSupportedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        return result;
	}
	
	public FullMappingEV getStateAsEV(AMapping state, List<Integer> exampleNums) {
		// get embedding vectors for source dataset
		String fileName = configuration.getSourceInfo().getEndpoint().split("datasets")[1];
		String dataPerson11Path = "src/main/resources/ConEx-vectors"+fileName;
		AMappingReader mappingReader;
    	mappingReader = new CSVMappingReader(dataPerson11Path);
    	HashMap<String, List<Double>> person11DataMap = new HashMap<String, List<Double>>();
        person11DataMap = ((CSVMappingReader) mappingReader).readEV();
        
        // get embedding vectors for target dataset
        String fileName2 = configuration.getTargetInfo().getEndpoint().split("datasets")[1];
        String dataPerson12Path = "src/main/resources/ConEx-vectors"+fileName2;
		AMappingReader mappingReader1;
    	mappingReader1 = new CSVMappingReader(dataPerson12Path);
    	HashMap<String, List<Double>> person12DataMap = new HashMap<String, List<Double>>();
        person12DataMap = ((CSVMappingReader) mappingReader1).readEV();
        
        List<List<Double>> stateEV = new ArrayList<List<Double>>();
        
        Integer num = 0;
        int counterExamplesNums = 0;
        if(exampleNums != null) {
        	num = exampleNums.get(0);
        }
        List<MappingEV> mEV = new ArrayList<MappingEV>();
        for (Entry<String, HashMap<String, Double>> entry : state.getMap().entrySet()) {
            for (Entry<String, Double> items : entry.getValue().entrySet()) {
            	List<Double> newList = new ArrayList<Double>();
            	newList.addAll(person11DataMap.get(entry.getKey()));
            	newList.addAll(person12DataMap.get(items.getKey()));
            	stateEV.add(newList);
            	MappingEV mappingEV = new MappingEV(num, entry.getKey(), items.getKey(), items.getValue());
            	mEV.add(mappingEV);
            	if(exampleNums != null) {
            		counterExamplesNums++;
            		if(exampleNums.size() < counterExamplesNums) {
            			num = exampleNums.get(counterExamplesNums);
            		}
            	}
            	num++;
            }
        }
        FullMappingEV m = new FullMappingEV(mEV, stateEV);
        return m;
	}
	
	public List<Integer> trainNNandGetExamples(List<List<Double>> stateEV, boolean isLastIterationOfAL) {
		// run python script with DQL 	        
        try {
            
            // any of the following work, these are just pseudo-examples
            
            // using exec(String) to invoke methods
	        interp.set("arg", stateEV);
	        interp.set("arg2", isLastIterationOfAL);
	        interp.set("WombatRLObject", this);
	        interp.exec("x = mainFun(arg, arg2)");
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
	
    public AMapping executeLS(LinkSpecification ls, ACache sCache, ACache tCache) {
        Rewriter rw = RewriterFactory.getDefaultRewriter();
        LinkSpecification rwLs = rw.rewrite(ls);
        IPlanner planner = ExecutionPlannerFactory.getPlanner(ExecutionPlannerType.DEFAULT, sCache, tCache);
        assert planner != null;
        ExecutionEngine engine = ExecutionEngineFactory.getEngine(ExecutionEngineType.DEFAULT, sCache, tCache,
                "?" + sourceVariable, "?" + targetVariable, 0, 1);
        assert engine != null;
        AMapping resultMap = engine.execute(rwLs, planner);
        return resultMap.getSubMap(ls.getThreshold());
    }
	
	public AMapping getLSandState(int N) {
		MLResults mlm;
//		if (firstStep) {
//			// using supervised learning
//	        mlm = this.learn(new PseudoFMeasure());
//	        firstStep = false;
//		} else {
			// using batch learning
			mlm = this.learn(this.trainingData);	
//		}
		
		
        LinkSpecification ls = mlm.getLinkSpecification();
//        ls.setThreshold(decisionBoundaryTheshold);
        logger.info("Learned: " + mlm.getLinkSpecification().getFullExpression() + " with threshold: " + mlm.getLinkSpecification().getThreshold());
  
//        AMapping results = executeLS(ls, sourceCache, targetCache); // getting mapping for the LS
        AMapping results = LSPipeline.execute(sourceCache, targetCache, ls);
        AMapping subMapping = results.getSubMap(decisionBoundaryTheshold - epsilon,decisionBoundaryTheshold + epsilon, this.trainingData, numberOfPairs, true);
        System.out.println("Results: "+results.size());
        System.out.println("Submapping: "+subMapping.size());
        return subMapping;
	}
	
	public AMapping getLSandNextState(double e) { // K - amount of examples nearest to the decision boundary
		// using batch learning
		 MLResults mlm = this.learn(this.trainingData);	
		
		// using supervised learning
//        MLResults mlm = this.learn(new PseudoFMeasure());
        
        logger.info("Learned: " + mlm.getLinkSpecification().getFullExpression() + " with threshold: " + mlm.getLinkSpecification().getThreshold());
        LinkSpecification ls = mlm.linkspec;
//        ls.setThreshold(decisionBoundaryTheshold);
	    AMapping results = LSPipeline.execute(sourceCache, targetCache, ls);
	    AMapping subMapping = results.getSubMap(decisionBoundaryTheshold - epsilon,decisionBoundaryTheshold + epsilon, this.trainingData, numberOfPairs, false);
	    System.out.println("Results: "+results.size());
        System.out.println("Submapping: "+subMapping.size());
        AMapping elem = null;
        try {
        	elem = subMapping.getRandomElementMap(experienceList);
        } catch(Exception er) {
        	System.out.println("Something went wrong.");
        }
        return elem;
	}
	
	public double countFMeasure() {
		String goldStandardDataFile = "src/main/resources/datasets/Abt-Buy/abt_buy_perfectMapping.csv";
        AMapping goldStandardDataMap = MappingFactory.createDefaultMapping();
        AMappingReader mappingReader;
        mappingReader = new CSVMappingReader(goldStandardDataFile);
        goldStandardDataMap = mappingReader.read();
		// calculate F-measure
        double newFMeasure = new FMeasure().calculate(this.trainingData, new GoldStandard(goldStandardDataMap), getBeta());
        this.currentReward = newFMeasure;
        logger.info("Current reward: "+this.currentReward);
		return newFMeasure;
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
}