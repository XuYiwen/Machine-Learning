package cs446.weka.classifiers.trees;

import weka.core.Instance;
import weka.core.Instances;

import java.util.Random;

/**
 * Created by XuYiwen on 9/22/16.
 */
public class Sgd {
    private Instances m_data;
    private int numInstances;
    private int numAttributes;
    private double[] m_weights;
    private double m_loss;

    private int m_maxEpochs = 1000;
    private double m_learningRate;
    private double m_threshold;

    public Sgd(double _learningRate, double _threshold){
        m_learningRate = _learningRate;
        m_threshold = _threshold;
    }

    public void buildClassifier(Instances data) throws Exception {
        data = new Instances(data);
        numInstances = data.numInstances();
        numAttributes = data.numAttributes() - 1;

        initialWeights();
        for (int e = 0; e < m_maxEpochs; e++) {
            m_loss = 0;
            for (int i = 0; i < numInstances; i++){
                double loss = predictLoss(data.instance(i));
                updateWeights(loss, data.instance(i));
                m_loss += loss * loss;
            }
            m_loss = 0.5 * m_loss;

            if (m_loss < m_threshold) {
                System.out.println("* Threshold reached at e = " + e);
                return;
            }

            if (e % 30 == 0) {
                System.out.println("Current Loss = " + m_loss);
            }
        }
    }

    public double evaluateClassifier(Instances data) throws Exception {
        double numCorrect = 0;
        double numIncorrect = 0;
        double accuracy;

        for (int i = 0; i < data.numInstances(); i++){
            double loss = predictLoss(data.instance(i));
            if (loss == 0) {
                numCorrect++;
            } else {
                numIncorrect++;
            }
        }
        accuracy = numCorrect / data.numInstances();
        System.out.println("Y/N/A = " + numCorrect + "/" + numIncorrect + "/" + data.numInstances());
        System.out.println("Accuracy(SGD): " + accuracy);

        return accuracy;
    }

    /**
     * Initialize weights with random double
     */
    private void initialWeights() {
        m_weights = new double[numAttributes];
        Random randomGenerator = new Random();
        for (int i = 0; i < m_weights.length; i++) {
            m_weights[i] = randomGenerator.nextDouble();
        }
    }

    private double predictLoss(Instance instance) {
        double[] attributes = getAttributes(instance);
        double label = getLabel(instance);

        double predictValue = dotProd(attributes, m_weights);
        double prediction = predictValue > 0 ? 0.0 : 1.0;

        return label - prediction;
    }

    private void updateWeights(double loss, Instance instance) {
        double[] attributes = getAttributes(instance);
        for (int i = 0; i < m_weights.length; i++) {
            m_weights[i] -= m_learningRate * loss * attributes[i];
        }
    }

    private double[] getAttributes(Instance instance){
        double[] attributesAndLabel = instance.toDoubleArray();
        double[] onlyAttributes = new double[attributesAndLabel.length-1];

        for (int i = 0; i < attributesAndLabel.length-1; i++) {
            onlyAttributes[i] = attributesAndLabel[i];
        }
        return onlyAttributes;
    }

    private double getLabel(Instance instance) {
        return instance.classValue();
    }


    private static double dotProd(double[] a, double[] b) {
        if (a.length != b.length) {
            System.err.println("Dot product error: dimension disagree.");
        }

        double dotProd = 0;
        for (int i = 0; i < a.length; i++) {
            dotProd += a[i] * b[i];
        }
        return dotProd;
    }

    public static void main(String[] args) {

    }


}
