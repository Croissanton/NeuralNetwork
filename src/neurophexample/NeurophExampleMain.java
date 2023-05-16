package neurophexample;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.util.TransferFunctionType;
import java.io.BufferedWriter;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.Random;

public class NeurophExampleMain {
    private static final int SAMPLES = 1000;
    private static final int GRID_SIZE = 100;
    private static final int NUM_EPOCHS = 1000; // After testing with higher numbers, we have found that around 1000 the error is already low enough
                                               // and from this point on it won't get much lower. We should stop here to avoid overfitting since we have a good enough model.
    private static final double SPACE = 2 * Math.PI / GRID_SIZE;



    public static void main(String[] args) {
        DataSet trainingSet = generateDataSet();
        DataSet validationSet = generateDataSet();
            try {
                PrintStream o = new PrintStream("output.txt");
                System.setOut(o);
            } catch (FileNotFoundException e) {
                throw new RuntimeException(e);
            }

        MultiLayerPerceptron neuralNetwork = new MultiLayerPerceptron(TransferFunctionType.TANH, 2, 5, 7, 5, 1);

        neuralNetwork.getLearningRule().setLearningRate(0.02); // After testing we ended up choosing 0.02 as the best learning rate with this number of epochs and neuralNetwork.
        neuralNetwork.getLearningRule().setMaxIterations(1); // Set to 1 so the learning is not iterative and is instead done with epochs.

        for (int i = 1; i <= NUM_EPOCHS; i++) {
            //learn() method is used to train the neural network with the training set.
            //It does a learning epoch each time it is called.
            neuralNetwork.getLearningRule().learn(trainingSet);

            double trainingError = calculateMeanSquaredError(neuralNetwork, trainingSet); //Error comparing the output of the neural network with the desired output of the training set.
            double validationError = calculateMeanSquaredError(neuralNetwork, validationSet); //Error comparing the output of the neural network with the desired output of the validation set.

            System.out.printf("Epoch: %d, Training Error: %.5f, Validation Error: %.5f\n", i, trainingError, validationError);
        }

        DataSet testSet = generateTestSamples();
        double testError = calculateMeanSquaredError(neuralNetwork, testSet);
        System.out.printf("Test Error: %.5f\n", testError);

        printToCsv(neuralNetwork, testSet);
    }

    private static DataSet generateDataSet() {
        DataSet dataSet = new DataSet(2, 1);
        Random random = new Random();

        for (int i = 0; i < NeurophExampleMain.SAMPLES; i++) {
            double x = (random.nextDouble() * 2 * Math.PI) - Math.PI;
            double y = (random.nextDouble() * 2 * Math.PI) - Math.PI;
            double f = Math.sin(x) * Math.cos(y);

            dataSet.add(new DataSetRow(new double[]{x, y}, new double[]{f}));
        }

        return dataSet;
    }

    private static DataSet generateTestSamples() {
        DataSet dataSet = new DataSet(2, 1);

        for (int i = 0; i < NeurophExampleMain.GRID_SIZE; i++) {
            for (int j = 0; j < NeurophExampleMain.GRID_SIZE; j++) {
                double x = -Math.PI + i * SPACE;
                double y = -Math.PI + j * SPACE;
                double f = Math.sin(x) * Math.cos(y);

                dataSet.add(new DataSetRow(new double[]{x, y}, new double[]{f}));
            }
        }

        return dataSet;
    }

    private static void printToCsv(NeuralNetwork<?> neuralNetwork, DataSet dataSet) {
        String[] headers = {"x", "y", "real", "desired"};
        String[][] data = new String[dataSet.size()][4];
        for (DataSetRow row : dataSet) {
            neuralNetwork.setInput(row.getInput());
            neuralNetwork.calculate();
            double[] networkOutput = neuralNetwork.getOutput();
            double output = row.getDesiredOutput()[0];
            data[dataSet.indexOf(row)] = new String[]{String.valueOf(row.getInput()[0]), String.valueOf(row.getInput()[1]), String.valueOf(networkOutput[0]), String.valueOf(output)};
        }

        String csvFile = "test.csv";
        try (BufferedWriter writer = new BufferedWriter(new java.io.FileWriter(csvFile))) {
            for (String header : headers) {
                writer.write(header);
                writer.write(",");
            }
            writer.newLine();
            for (String[] row : data) {
                for (String col : row) {
                    writer.write(col);
                    writer.write(",");
                }
                writer.newLine();
            }
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }

    }

    private static double calculateMeanSquaredError(NeuralNetwork<?> neuralNetwork, DataSet dataSet) {
        double mse = 0.0;

        for (DataSetRow row : dataSet) {
            neuralNetwork.setInput(row.getInput());
            neuralNetwork.calculate();
            double[] networkOutput = neuralNetwork.getOutput();

            double output = row.getDesiredOutput()[0];
            double diff = output - networkOutput[0];
            mse += diff * diff;
        }

        return mse / dataSet.size();
    }
}