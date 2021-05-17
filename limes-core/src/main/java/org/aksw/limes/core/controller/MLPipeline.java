package org.aksw.limes.core.controller;

import java.util.List;
import org.aksw.limes.core.evaluation.evaluator.EvaluatorFactory;
import org.aksw.limes.core.evaluation.evaluator.EvaluatorType;
import org.aksw.limes.core.evaluation.qualititativeMeasures.PseudoFMeasure;
import org.aksw.limes.core.exceptions.UnsupportedMLImplementationException;
import org.aksw.limes.core.io.cache.ACache;
import org.aksw.limes.core.io.config.Configuration;
import org.aksw.limes.core.io.mapping.AMapping;
import org.aksw.limes.core.io.mapping.MappingFactory;
import org.aksw.limes.core.io.mapping.reader.AMappingReader;
import org.aksw.limes.core.io.mapping.reader.CSVMappingReader;
import org.aksw.limes.core.io.mapping.reader.RDFMappingReader;
import org.aksw.limes.core.ml.algorithm.ACoreMLAlgorithm;
import org.aksw.limes.core.ml.algorithm.ActiveMLAlgorithm;
import org.aksw.limes.core.ml.algorithm.LearningParameter;
import org.aksw.limes.core.ml.algorithm.MLAlgorithmFactory;
import org.aksw.limes.core.ml.algorithm.MLImplementationType;
import org.aksw.limes.core.ml.algorithm.MLResults;
import org.aksw.limes.core.ml.algorithm.SupervisedMLAlgorithm;
import org.aksw.limes.core.ml.algorithm.UnsupervisedMLAlgorithm;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Execution pipeline for generating mappings using ML.
 * Provides overloaded convenience methods.
 *
 * @author Kevin Dreßler
 */
public class MLPipeline {

    public static final Logger logger = LoggerFactory.getLogger(MLPipeline.class);
    
    // temporary for oracle feedback
    private static AMapping oracleFeedback(AMapping predictionMapping, AMapping referenceMapping) {
        AMapping result = MappingFactory.createDefaultMapping();

        for(String s : predictionMapping.getMap().keySet()){
            for(String t : predictionMapping.getMap().get(s).keySet()){
                if(referenceMapping.contains(s, t)){
                    result.add(s, t, predictionMapping.getMap().get(s).get(t));
                }
            }
        }
        return result;
    }
    
    public static AMapping getDataMapFromGoldStandard() {
    	String goldStandardDataFile = "src/main/resources/datasets/Abt-Buy/abt_buy_perfectMapping.csv";
        AMapping goldStandardDataMap = MappingFactory.createDefaultMapping();
        AMappingReader mappingReader;
        mappingReader = new CSVMappingReader(goldStandardDataFile);
        goldStandardDataMap = mappingReader.read();
		return goldStandardDataMap;
    }
    
    public static AMapping getRandomDataMapFromGoldStandard() {
    	AMapping goldStandardDataMap = getDataMapFromGoldStandard();
    	int size = 150;
    	AMapping randomDataMap = goldStandardDataMap.getRandomMap(size);
    	return randomDataMap;
    }

    public static AMapping execute(
            ACache source,
            ACache target,
            Configuration configuration,
            String mlAlgorithmName,
            MLImplementationType mlImplementationType,
            List<LearningParameter> learningParameters,
            String trainingDataFile,
            EvaluatorType pfmType,
            int maxIt,
            ActiveLearningOracle oracle
    ) throws UnsupportedMLImplementationException {
        Class<? extends ACoreMLAlgorithm> clazz = MLAlgorithmFactory.getAlgorithmType(mlAlgorithmName);
        MLResults mlm;
        AMapping trainingDataMap = MappingFactory.createDefaultMapping();
        if (mlImplementationType == MLImplementationType.SUPERVISED_BATCH || mlImplementationType == MLImplementationType.SUPERVISED_ACTIVE){
        	AMappingReader mappingReader;
        	if(trainingDataFile.endsWith(".csv")){
        		mappingReader = new CSVMappingReader(trainingDataFile);
        	}else{
        		mappingReader = new RDFMappingReader(trainingDataFile);
        	}
            trainingDataMap = mappingReader.read();
        }
        AMapping randomTrainingDataMap = MappingFactory.createDefaultMapping();

        switch (mlImplementationType) {
            case SUPERVISED_BATCH:
                SupervisedMLAlgorithm mls = new SupervisedMLAlgorithm(clazz);
                mls.init(learningParameters, source, target);
                mls.getMl().setConfiguration(configuration);
                mlm = mls.learn(trainingDataMap);
                logger.info("Learned: " + mlm.getLinkSpecification().getFullExpression() + " with threshold: " + mlm.getLinkSpecification().getThreshold());
                return mls.predict(source, target, mlm);
            case SUPERVISED_ACTIVE:
            	// for active learning, need to reiterate and prompt the user for evaluation of examples:
                //            boolean stopLearning = false;
                ActiveMLAlgorithm mla = new ActiveMLAlgorithm(clazz);
                mla.init(learningParameters, source, target);
                mla.getMl().setConfiguration(configuration);
                AMapping nextExamplesMapping;
                if(mlAlgorithmName.equals("WOMBAT Simple RL")){
                	// get random trainingDataMap from the gold standard
                	randomTrainingDataMap = getRandomDataMapFromGoldStandard();
//                	randomTrainingDataMap = MappingFactory.createDefaultMapping(); // if pseudoFMeasure
                	mlm = mla.learn(randomTrainingDataMap); // trainingDataMap
                } else {
                	mlm = mla.activeLearn();	
                }
                
                if(mlAlgorithmName.equals("WOMBAT Simple RL")){ // RL version
                	boolean finished = false;
                	for(int iter=0; iter<100 && !finished; iter++) {
                		logger.info("Iteration:" + iter);
                    	nextExamplesMapping = null;
                    	oracle = new AsynchronousServerOracle();
                        while (!oracle.isStopped()) {
                        	nextExamplesMapping = mla.getNextExamples(maxIt);
                            if (nextExamplesMapping == null) {
                                oracle.stop();
                                finished = true;
                                break;
                            }
                            if (nextExamplesMapping.getMap().isEmpty()) {
                                oracle.stop();
                                break;
                            }
                            logger.info(nextExamplesMapping.toString());
                         // instead feedback from user - later uncomment
//                            ActiveLearningExamples activeLearningExamples = new ActiveLearningExamples(nextExamplesMapping, source, target);
//                            AMapping classify = oracle.classify(activeLearningExamples);
                            
                            // temporary for oracle feedback
//                            String goldStandardDataFile = "src/main/resources/datasets/Abt-Buy/abt_buy_perfectMapping.csv";
//                            AMapping goldStandardDataMap = MappingFactory.createDefaultMapping();
//                            AMappingReader mappingReader;
//                            mappingReader = new CSVMappingReader(goldStandardDataFile);
//                            goldStandardDataMap = mappingReader.read();
                            AMapping goldStandardDataMap = getDataMapFromGoldStandard();
                            AMapping classify = oracleFeedback(nextExamplesMapping,goldStandardDataMap);
                            /////////
                            logger.info(classify.toString());
                            mlm = mla.activeLearn(classify);
                            if(classify.size() != 0) {
                            	AMapping trainingDataMap1 = MappingFactory.createDefaultMapping();//trainingDataMap;
                            	classify.getMap().forEach((k,v) -> {
                            		trainingDataMap1.add(k, v);
                            	});
                            	randomTrainingDataMap.getMap().forEach((k,v) -> {
                            		trainingDataMap1.add(k, v);
                            	});
                            	randomTrainingDataMap = trainingDataMap1;
                            }
                            logger.info("Current TestK size: "+randomTrainingDataMap.size());
                        }
                        logger.info("Learned: " + mlm.getLinkSpecification().getFullExpression() + " with threshold: " + mlm.getLinkSpecification().getThreshold());
                        
                    	}
                } else { // old version
                        while (!oracle.isStopped()) {
                        	nextExamplesMapping = mla.getNextExamples(maxIt);
                            if (nextExamplesMapping.getMap().isEmpty()) {
                                oracle.stop();
                                break;
                            }
                            logger.info(nextExamplesMapping.toString());
                         // instead feedback from user - later uncomment
                            ActiveLearningExamples activeLearningExamples = new ActiveLearningExamples(nextExamplesMapping, source, target);
                            AMapping classify = oracle.classify(activeLearningExamples);
                            
                            logger.info(classify.toString());
                            mlm = mla.activeLearn(classify);
                        }
                        logger.info("Learned: " + mlm.getLinkSpecification().getFullExpression() + " with threshold: " + mlm.getLinkSpecification().getThreshold());
                }

            	return mla.predict(source, target, mlm);
            case UNSUPERVISED:
                UnsupervisedMLAlgorithm mlu = new UnsupervisedMLAlgorithm(clazz);
                mlu.init(learningParameters, source, target);
                mlu.getMl().setConfiguration(configuration);
                PseudoFMeasure pfm = null;
                if(pfmType != null){
                    pfm = (PseudoFMeasure) EvaluatorFactory.create(pfmType);
                }
                mlm = mlu.learn(pfm);
                logger.info("Learned: " + mlm.getLinkSpecification().getFullExpression() + " with threshold: " + mlm.getLinkSpecification().getThreshold());
                return mlu.predict(source, target, mlm);
            default:
                throw new UnsupportedMLImplementationException(clazz.getName());
        }
    }
}
