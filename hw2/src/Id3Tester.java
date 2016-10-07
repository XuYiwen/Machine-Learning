package cs446.homework2;

import cs446.weka.classifiers.trees.Id3;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.PrintStream;
import java.util.Enumeration;

/**
 * Created by XuYiwen on 9/22/16.
 */
public class Id3Tester {
    private final String inputPrefix;
    private final String logFileName;
    private final int numFolds;

    public Instances trainData;
    public Instances testData;

    public Id3Tester(int _numFolds, String _inputPrefix, String _logFileName) {
        inputPrefix = _inputPrefix;
        logFileName = _logFileName;
        numFolds = _numFolds;
    }

    private void distributeData(int testFold) throws Exception {
        // initialize training data and test data
        trainData = new Instances(new FileReader(new File(inputPrefix + 1)));
        trainData.delete();
        testData = new Instances(new FileReader(new File(inputPrefix + 1)));
        testData.delete();

        // distribute each fold to training data or testing data.
        for (int i = 1; i <= numFolds; i++) {
            Instances currentFold = new Instances(new FileReader(new File(inputPrefix + i)));
            if (i != testFold) {
                trainData = copyInstances(trainData, currentFold);
            } else {
                testData = copyInstances(testData, currentFold);
                System.out.println(">> Test Fold = " + i);
            }
        }

        System.out.println(">> Training Data: " + trainData.numInstances());
        System.out.println(">> Testing Data: " + testData.numInstances());

        trainData.setClassIndex(trainData.numAttributes() - 1);
        testData.setClassIndex(testData.numAttributes() - 1);
    }

    private static Instances copyInstances(Instances to, Instances from){
        Enumeration fromEnum = from.enumerateInstances();

        while (fromEnum.hasMoreElements()) {
            to.add((Instance) fromEnum.nextElement());
        }
        return to;
    }

    public static void main(String[] args) throws Exception {

        if (args.length != 3) {
            System.err.println("Usage: Id3Tester numFolds arff-file-prefix output-prefix");
            System.exit(-1);
        }

        Id3Tester tester = new Id3Tester(
                Integer.parseInt(args[0]), args[1], args[2]);

        // Set output log
        FileOutputStream fis = new FileOutputStream(new File(args[2]));
        PrintStream out = new PrintStream(fis);
        System.setOut(out);

        double averRate = 0;
        for (int n = 1; n <= tester.numFolds; n++) {
            // Split dataset
            tester.distributeData(n);

            // Build Classifier
            Id3 classifier = new Id3();
            classifier.setMaxDepth(8);

            // Train
            classifier.buildClassifier(tester.trainData);

            // Print the classfier
            System.out.println(classifier);
            System.out.println();

            // Evaluate on the test set
            Evaluation evaluation = new Evaluation(tester.testData);
            evaluation.evaluateModel(classifier, tester.testData);
            System.out.println(evaluation.toSummaryString());

            // Record for average
            averRate += evaluation.pctCorrect();
        }
        averRate /= tester.numFolds;
        System.out.println("\n>> Average Accuracy = " + averRate);
    }
}

