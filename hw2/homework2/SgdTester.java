package cs446.homework2;

import cs446.weka.classifiers.trees.Id3;
import cs446.weka.classifiers.trees.Sgd;
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
public class SgdTester {
    private final String inputPrefix;
    private final String logFileName;
    private final int numFolds;

    public Instances validateData;
    public Instances trainData;
    public Instances testData;

    public SgdTester(int _numFolds, String _inputPrefix, String _logFileName) {
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
        validateData = new Instances(new FileReader(new File(inputPrefix + 1)));
        validateData.delete();

        // distribute each fold to training data or testing data.
        for (int i = 1; i <= numFolds; i++) {
            Instances currentFold = new Instances(new FileReader(new File(inputPrefix + i)));

            if (i != testFold) {
                if (trainData.numInstances() == 0) {
                    // Let validate Data to be the first fold in training data.
                    validateData = copyInstances(validateData, currentFold);
                    System.out.println(">> Validate Fold = " + i);
                }
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
        validateData.setClassIndex(validateData.numAttributes() - 1);
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
            System.err.println("Usage: SgdTester numFolds arff-file-prefix output-prefix");
            System.exit(-1);
        }

        SgdTester tester = new SgdTester(
                Integer.parseInt(args[0]), args[1], args[2]);

        boolean tuning = false;
        if (tuning) {
            // Tuning
            tester.distributeData(1); // change this
            Sgd classifier = new Sgd(0.1, 0.0001);
            classifier.buildClassifier(tester.validateData);
            System.out.println();
            classifier.evaluateClassifier(tester.testData);
        } else {
            // Set output log
            FileOutputStream fis = new FileOutputStream(new File(args[2]));
            PrintStream out = new PrintStream(fis);
            System.setOut(out);

            double averRate = 0;
            for (int n = 1; n <= tester.numFolds; n++) {
                // Split dataset
                tester.distributeData(n);

                // Build Classifier
                Sgd classifier = new Sgd(0.1, 0.0001);

                // Train
                classifier.buildClassifier(tester.trainData);

                // Record for average
                System.out.println();
                averRate += classifier.evaluateClassifier(tester.testData);
                System.out.println();
            }
            averRate /= tester.numFolds;
            System.out.println("\n>> Average Accuracy = " + averRate);
        }
    }
}

