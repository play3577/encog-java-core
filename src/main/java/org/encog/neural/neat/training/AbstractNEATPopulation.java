package org.encog.neural.neat.training;

import org.encog.engine.network.activation.ActivationFunction;
import org.encog.ml.MLError;
import org.encog.ml.MLRegression;
import org.encog.ml.ea.genome.GenomeFactory;
import org.encog.ml.ea.population.BasicPopulation;
import org.encog.util.obj.ChooseObject;

import java.io.Serializable;

/**
 * Created by hardwiwill on 28/11/16.
 */
public abstract class AbstractNEATPopulation extends BasicPopulation implements Serializable,
        MLError, MLRegression {

    private ChooseObject<Object> activationFunctions;

    public AbstractNEATPopulation(){}

    public AbstractNEATPopulation(int populationSize, GenomeFactory genomeFactory) {
        super(populationSize, genomeFactory);
    }

    public abstract long assignGeneID();
    public abstract long assignInnovationID();

    public abstract int getActivationCycles();
    public abstract double getWeightRange();

    public abstract ChooseObject<ActivationFunction> getActivationFunctions();

    public abstract NEATInnovationList getInnovations();

}
