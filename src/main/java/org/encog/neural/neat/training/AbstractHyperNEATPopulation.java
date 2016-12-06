package org.encog.neural.neat.training;

import org.encog.engine.network.activation.ActivationFunction;
import org.encog.ml.ea.genome.GenomeFactory;
import org.encog.neural.hyperneat.substrate.Substrate;

/**
 * Created by hardwiwill on 30/11/16.
 */
public abstract class AbstractHyperNEATPopulation extends AbstractNEATPopulation {

    public AbstractHyperNEATPopulation() {}

    public AbstractHyperNEATPopulation(int populationSize, GenomeFactory genomeFactory) {
        super(populationSize, genomeFactory);
    }

    public abstract ActivationFunction getHyperNEATNNActivationFunction();

    public abstract double getCPPNMinWeight();

    public abstract double getHyperNEATNNWeightRange();

    public abstract Substrate getSubstrate();
}
