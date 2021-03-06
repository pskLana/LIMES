package org.aksw.limes.core.execution.engine;

import static org.junit.Assert.assertTrue;

import org.aksw.limes.core.execution.engine.ExecutionEngineFactory.ExecutionEngineType;
import org.aksw.limes.core.execution.engine.partialrecallengine.PartialRecallExecutionEngine;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;


public class ExecutionEngineFactoryTest {


    public ExecutionEngineFactoryTest() {
    }


    @Before
    public void setUp() {

    }

    @After
    public void tearDown() {

    }

    @Test
    public void testEqualDefault() {
        ExecutionEngine engine = ExecutionEngineFactory.getEngine(ExecutionEngineType.DEFAULT, null, null, null, null, 0, 1.0);
        assertTrue(engine instanceof SimpleExecutionEngine);
    }

    @Test
    public void testEqualLiger() {
        ExecutionEngine engine = ExecutionEngineFactory.getEngine(ExecutionEngineType.PARTIAL_RECALL, null, null, null, null, 0, 1.0);
        assertTrue(engine instanceof PartialRecallExecutionEngine);
    }
}
