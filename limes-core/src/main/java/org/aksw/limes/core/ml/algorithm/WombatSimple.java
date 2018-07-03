package org.aksw.limes.core.ml.algorithm;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.TreeSet;

import org.aksw.limes.core.datastrutures.LogicOperator;
import org.aksw.limes.core.datastrutures.Tree;
import org.aksw.limes.core.evaluation.qualititativeMeasures.PseudoFMeasure;
import org.aksw.limes.core.exceptions.UnsupportedMLImplementationException;
import org.aksw.limes.core.io.cache.ACache;
import org.aksw.limes.core.io.ls.LinkSpecification;
import org.aksw.limes.core.io.mapping.AMapping;
import org.aksw.limes.core.io.mapping.MappingFactory;
import org.aksw.limes.core.io.mapping.MappingFactory.MappingType;
import org.aksw.limes.core.measures.mapper.CrispSetOperations;
import org.aksw.limes.core.measures.mapper.MappingOperations;
import org.aksw.limes.core.ml.algorithm.classifier.ExtendedClassifier;
import org.aksw.limes.core.ml.algorithm.eagle.util.PropertyMapping;
import org.aksw.limes.core.ml.algorithm.fptld.nodes.InternalNode;
import org.aksw.limes.core.ml.algorithm.wombat.AWombat;
import org.aksw.limes.core.ml.algorithm.wombat.LinkEntropy;
import org.aksw.limes.core.ml.algorithm.wombat.RefinementNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Simple implementation of the Wombat algorithm
 * Fast implementation, that is not complete
 *
 * @author Mohamed Sherif (sherif@informatik.uni-leipzig.de)
 * @version Jun 7, 2016
 */
public class WombatSimple extends AWombat {
	protected static Logger logger = LoggerFactory.getLogger(WombatSimple.class);
	protected static final String ALGORITHM_NAME = "Wombat Simple";
	protected int activeLearningRate = 3;
	protected RefinementNode bestSolutionNode = null;
	protected List<ExtendedClassifier> classifiers = null;
	protected int iterationNr = 0;
	public PropertyMapping propMap;
	private LogicOperator[] availableOperators;


	/**
	 * WombatSimple constructor.
	 */
	protected WombatSimple() {
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
		bestSolutionNode = null;
		classifiers = null;
		iterationNr = 0;
	}

	@Override
	protected MLResults learn(AMapping trainingData) {
		this.trainingData = trainingData;
		fillSampleSourceTargetCaches(trainingData);
		sourceCache = sourceSample;
		targetCache = targetSample;
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
			pseudoFMeasure = pfm;
		}else{ // use default PFM
			pseudoFMeasure = new PseudoFMeasure();
		}
		isUnsupervised = true;
		return learn();
	}

	@Override
	protected AMapping predict(ACache source, ACache target, MLResults mlModel) {
		LinkSpecification ls = mlModel.getLinkSpecification();
		return getPredictions(ls, source, target);
	}

	@Override
	protected boolean supports(MLImplementationType mlType) {
		return mlType == MLImplementationType.SUPERVISED_BATCH ||
				mlType == MLImplementationType.UNSUPERVISED    ||
				mlType == MLImplementationType.SUPERVISED_ACTIVE;
	}

	@Override
	protected AMapping getNextExamples(int size) throws UnsupportedMLImplementationException {
		List<RefinementNode> bestNodes = getBestKNodes(refinementTreeRoot, activeLearningRate);
		AMapping intersectionMapping = MappingFactory.createDefaultMapping();
		AMapping unionMapping = MappingFactory.createDefaultMapping();

		for(RefinementNode sn : bestNodes){
			intersectionMapping = CrispSetOperations.INSTANCE.intersection(intersectionMapping, sn.getMapping());
			unionMapping = CrispSetOperations.INSTANCE.union(unionMapping, sn.getMapping());
		}
		AMapping posEntropyMapping = CrispSetOperations.INSTANCE.difference(unionMapping, intersectionMapping);

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
			highestEntropyLinks.add(itr.next());
			i++;
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
	protected MLResults activeLearn(AMapping oracleMapping) throws UnsupportedMLImplementationException {
		isUnsupervised = false;
		trainingData = CrispSetOperations.INSTANCE.union(trainingData, oracleMapping);
		updateScores(refinementTreeRoot);
		bestSolutionNode = findBestSolution();
		String bestMetricExpr = bestSolutionNode.getMetricExpression();
		double threshold = Double.parseDouble(bestMetricExpr.substring(bestMetricExpr.lastIndexOf("|") + 1, bestMetricExpr.length()));
		AMapping bestMapping = bestSolutionNode.getMapping();
		LinkSpecification bestLS = new LinkSpecification(bestMetricExpr, threshold);
		double bestFMeasure = bestSolutionNode.getFMeasure();
		return new MLResults(bestLS, bestMapping, bestFMeasure, null);
	}



	/**
	 * update precision, recall and F-Measure of the refinement tree r
	 * based on either training data or PFM
	 *
	 * @param r refinement tree
	 */
	protected void updateScores(Tree<RefinementNode> r) {
		if (r.getchildren() == null || r.getchildren().size() == 0) {
			r.getValue().setfMeasure(fMeasure(r.getValue().getMapping()));
			r.getValue().setPrecision(precision(r.getValue().getMapping()));
			r.getValue().setRecall(recall(r.getValue().getMapping()));
			return;
		}
		for (Tree<RefinementNode> child : r.getchildren()) {
			if (child.getValue().getFMeasure() >= 0) {
				r.getValue().setfMeasure(fMeasure(r.getValue().getMapping()));
				r.getValue().setPrecision(precision(r.getValue().getMapping()));
				r.getValue().setRecall(recall(r.getValue().getMapping()));
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
		Tree<RefinementNode> mostPromisingNode = getMostPromisingNode(refinementTreeRoot, getOverAllPenaltyWeight() );
		logger.debug("Most promising node: " + mostPromisingNode.getValue());
		iterationNr++;
		while (mostPromisingNode.getValue().getFMeasure() < getMaxFitnessThreshold()
				&& refinementTreeRoot.size() <= getMaxRefinmentTreeSize()
				&& iterationNr <= getMaxIterationNumber()) {
			iterationNr++;
			mostPromisingNode = expandNode(mostPromisingNode);
			mostPromisingNode = getMostPromisingNode(refinementTreeRoot, getOverAllPenaltyWeight());
			if (mostPromisingNode.getValue().getFMeasure() == -Double.MAX_VALUE) {
				break; // no better solution can be found
			}
			logger.debug("Most promising node: " + mostPromisingNode.getValue());
		}
		RefinementNode bestSolution = getMostPromisingNode(refinementTreeRoot, 0).getValue();
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
	 * @return The input tree node after expansion
	 * @author sherif
	 */
	private Tree<RefinementNode> expandNode(Tree<RefinementNode> node) {
		AMapping map = MappingFactory.createDefaultMapping();
		for (ExtendedClassifier c : classifiers) {
			for (LogicOperator op : availableOperators) {
				if (!node.getValue().getMetricExpression().equals(c.getMetricExpression())) { // do not create the same metricExpression again
					map = MappingOperations.performOperation(node.getValue().getMapping(), c.getMapping(), op);
					String metricExpr = InternalNode.opToExpMapping.get(op) + "("
							+ node.getValue().getMetricExpression() + "," + c.getMetricExpression() + ")|0";
					RefinementNode child = createNode(map, metricExpr);
					node.addChild(new Tree<>(child));
				}
			}

		}
		if (isVerbose()) {
			refinementTreeRoot.print();
		}
		return node;
	}



	/**
	 * initiate the refinement tree as a root node  with set of
	 * children nodes containing all initial classifiers
	 *
	 */
	protected void createRefinementTreeRoot() {
		RefinementNode initialNode = new RefinementNode(-Double.MAX_VALUE, MappingFactory.createMapping(MappingType.DEFAULT), "");
		refinementTreeRoot = new Tree<>(null, initialNode, null);
		for (ExtendedClassifier c : classifiers) {
			RefinementNode n = new RefinementNode(c.getfMeasure(), c.getMapping(), c.getMetricExpression());
			refinementTreeRoot.addChild(new Tree<>(refinementTreeRoot, n, null));
		}
		if (isVerbose()) {
			refinementTreeRoot.print();
		}
	}

	public LogicOperator[] getAvailableOperators() {
		return availableOperators;
	}

	public void setAvailableOperators(LogicOperator[] availableOperators) {
		this.availableOperators = availableOperators;
	}


}
