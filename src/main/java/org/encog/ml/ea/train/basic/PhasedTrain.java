package org.encog.ml.ea.train.basic;

import org.encog.ml.CalculateScore;
import org.encog.ml.ea.opp.EvolutionaryOperator;
import org.encog.ml.ea.opp.OperationList;
import org.encog.ml.ea.population.Population;

/**
 * Created by Will on 20/09/2016.
 */
public class PhasedTrain extends TrainEA {
    private int phase = 0;
    private final int maxPhases = 2;
    private OperationList phaseAOps = new OperationList();
    private OperationList phaseBOps = new OperationList();

    public PhasedTrain(Population thePopulation, CalculateScore theScoreFunction) {
        super(thePopulation, theScoreFunction);
    }

    public void switchPhase() {
        phase = phase % (phase + 1);
        System.out.println("Phase changed to : " + phase);

        // todo: remove
//        getOperators().remove();
    }

    public void addPhaseAOp(double prob, EvolutionaryOperator op) {
        phaseAOps.add(prob, op);
        op.init(this);
    }

    public void addPhaseBOp(double prob, EvolutionaryOperator op) {
        phaseBOps.add(prob, op);
        op.init(this);
    }
}
