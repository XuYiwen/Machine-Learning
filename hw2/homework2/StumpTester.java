package cs446.homework2;

import cs446.weka.classifiers.trees.Id3;
import cs446.weka.classifiers.trees.Sgd;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.PrintStream;
import java.util.HashMap;
import java.util.Random;

/**
 * Created by XuYiwen on 9/22/16.
 */
public class StumpTester {
    private static HashMap<String, Attribute> attributesMap = new HashMap<>();
    private final String inputFileName;
    private final String logFileName;
    private final int numFolds;
    private final int numId3 = 100;

    private Id3[] stumps = new Id3[numId3];
    private Instances rawFeatureData;
    private Instances stumpFeatureData;

    public StumpTester(int _numFolds, String _inputFileName, String _logFileName) throws Exception {
        inputFileName = _inputFileName;
        logFileName = _logFileName;
        numFolds = _numFolds;

        // Load raw feature data
        rawFeatureData = new Instances(new FileReader(new File(inputFileName)));
        rawFeatureData.setClassIndex(rawFeatureData.numAttributes() - 1);
    }

    public void createStumpFeatureData() throws Exception{
        // generate id3 stumps
        for (int i = 0; i < numId3; i++) {
            rawFeatureData.randomize(new Random());
            Instances stumpData = rawFeatureData.trainCV(2, 0);
            stumps[i] = new Id3();
            stumps[i].setMaxDepth(4);
            stumps[i].buildClassifier(stumpData);
        }

        // Make Feature of the stumps
        FastVector zeroOne = new FastVector(2);
        zeroOne.addElement("1");
        zeroOne.addElement("0");
        FastVector labels = new FastVector(2);
        labels.addElement("+");
        labels.addElement("-");

        FastVector attributes = new FastVector();
        for (int i = 0; i < numId3; i++) {
            Attribute attr = new Attribute("stump" + i, zeroOne);
            attributesMap.put("stump" + i, attr);
            attributes.addElement(attr);
        }
        Attribute classLabel = new Attribute("Class", labels);
        attributes.addElement(classLabel);

        // Extract Features from id3
        stumpFeatureData = new Instances("Id3 Stumps Features", attributes, 0);
        stumpFeatureData.setClass(classLabel);
        for (int i = 0; i < rawFeatureData.numInstances(); i++) {
            Instance rawInstance = rawFeatureData.instance(i);

            // Use the prediction of each stump as new feature
            Instance featureInstance = new Instance(numId3 + 1);
            featureInstance.setDataset(stumpFeatureData);
            for (int j = 0; j < numId3; j++){
                double prediction = stumps[j].classifyInstance(rawInstance);
                String value = prediction == 0.0 ? "1": "0";
                featureInstance.setValue(attributesMap.get("stump" + j), value);
            }
            featureInstance.setClassValue((rawInstance.classValue() == 0.0 ? "+" : "-"));
            stumpFeatureData.add(featureInstance);
        }
    }

    public static void main(String[] args) throws Exception {

        if (args.length != 3) {
            System.err.println("Usage: StumpTester numFolds arff-file output-prefix");
            System.exit(-1);
        }

        StumpTester tester = new StumpTester(
                Integer.parseInt(args[0]), args[1], args[2]);
        tester.createStumpFeatureData();

        // Set output log
        FileOutputStream fis = new FileOutputStream(new File(tester.logFileName));
        PrintStream out = new PrintStream(fis);
        System.setOut(out);

        double averRate = 0;
        for (int n = 0; n < tester.numFolds; n++) {
            // Split dataset
            Instances train = tester.stumpFeatureData.trainCV(tester.numFolds, n);
            Instances test = tester.stumpFeatureData.trainCV(tester.numFolds, n);

            // Build Classifier
            Sgd classifier = new Sgd(0.1, 0.0001);

            // Train
            classifier.buildClassifier(train);

            // Evaluate on the test set
            // Record for average
            System.out.println();
            averRate += classifier.evaluateClassifier(test);
            System.out.println();
        }
        averRate /= tester.numFolds;
        System.out.println("\n>> Average Accuracy = " + averRate);
    }
}

