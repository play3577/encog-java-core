package org.encog.neural.neat.training;

import org.encog.ml.ea.genome.BasicGenome;
import org.encog.ml.ea.genome.Genome;
import org.encog.util.Format;

import java.io.Serializable;
import java.util.List;

/**
 * Created by hardwiwill on 28/11/16.
 */
public abstract class NEATGenome extends BasicGenome implements Cloneable, Serializable {

    /**
     * Serial id.
     */
    private static final long serialVersionUID = 1L;

    /**
     * The number of outputs.
     */
    protected int outputCount;

    protected int inputCount;

    protected int networkDepth;

    public NEATGenome() {}

    public NEATGenome(int inputCount, int outputCount) {
        this.inputCount = inputCount;
        this.outputCount = outputCount;
    }

    /**
     * @return The number of input neurons.
     */
    public int getInputCount() {
        return inputCount;
    }

    /**
     * @return The network depth.
     */
    public int getNetworkDepth() {
        return this.networkDepth;
    }

    /**
     * @return The number of genes in the links chromosome.
     */
    public abstract int getNumGenes();

    /**
     * @return The output count.
     */
    public int getOutputCount() {
        return outputCount;
    }

    /**
     * @param networkDepth
     *            the networkDepth to set
     */
    public void setNetworkDepth(final int networkDepth) {
        this.networkDepth = networkDepth;
    }

    /**
     * Sort the genes.
     */
    public abstract void sortGenes();

    /**
     * @return the linksChromosome
     */
    public abstract List<NEATLinkGene> getLinksChromosome();

    /**
     * @return the neuronsChromosome
     */
    public abstract List<NEATNeuronGene> getNeuronsChromosome();

    /**
     * @param inputCount
     *            the inputCount to set
     */
    public void setInputCount(int inputCount) {
        this.inputCount = inputCount;
    }

    /**
     * @param outputCount
     *            the outputCount to set
     */
    public void setOutputCount(int outputCount) {
        this.outputCount = outputCount;
    }

    /**
     * Validate the structure of this genome.
     */
    public abstract void validate();

    /**
     * {@inheritDoc}
     */
    public void copy(Genome source) {
        throw new UnsupportedOperationException();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public abstract int size();

    /**
     * Find the neuron with the specified nodeID.
     *
     * @param nodeID
     *            The nodeID to look for.
     * @return The neuron, if found, otherwise null.
     */
    public abstract NEATNeuronGene findNeuron(long nodeID);

}
