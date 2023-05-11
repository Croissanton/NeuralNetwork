package neurophexample;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.util.TransferFunctionType;
import java.io.BufferedWriter;

import java.util.Random;


public class NeurophExampleMain {
    private static final int SAMPLES = 1000;
    private static final int GRID_SIZE = 100;
    private static final int NUM_EPOCHS = 500; // After testing with higher numbers, we have found that 100 is where the square error reaches its minimum.
    private static final double SPACE = 2 * Math.PI / GRID_SIZE;

    public static void main(String[] args) {
        DataSet trainingSet = generateDataSet(SAMPLES);
        DataSet validationSet = generateDataSet(SAMPLES);

        MultiLayerPerceptron neuralNetwork = new MultiLayerPerceptron(TransferFunctionType.TANH, 2, 5, 7, 5, 1);
        neuralNetwork.getLearningRule().setLearningRate(0.1);
        neuralNetwork.getLearningRule().setMaxError(0.01);

        for (int i = 1; i <= NUM_EPOCHS; i++) {
            neuralNetwork.learn(trainingSet);

            double trainingError = calculateMeanSquaredError(neuralNetwork, trainingSet);
            double validationError = calculateMeanSquaredError(neuralNetwork, validationSet);

            System.out.printf("Epoch: %d, Training Error: %.5f, Validation Error: %.5f\n", i, trainingError, validationError);
        }

        DataSet testSet = generateTestSamples(GRID_SIZE);
        double testError = calculateMeanSquaredError(neuralNetwork, testSet);
        System.out.printf("Test Error: %.5f\n", testError);

        printToCsv(neuralNetwork, testSet);
    }

    private static DataSet generateDataSet(int size) {
        DataSet dataSet = new DataSet(2, 1);
        Random random = new Random();

        for (int i = 0; i < size; i++) {
            double x = (random.nextDouble() * 2 * Math.PI) - Math.PI;
            double y = (random.nextDouble() * 2 * Math.PI) - Math.PI;
            double f = Math.sin(x) * Math.cos(y);

            dataSet.add(new DataSetRow(new double[]{x, y}, new double[]{f}));
        }

        return dataSet;
    }

    private static DataSet generateTestSamples(int gridSize) {
        DataSet dataSet = new DataSet(2, 1);

        for (int i = 0; i < gridSize; i++) {
            for (int j = 0; j < gridSize; j++) {
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
//            System.out.printf("x: %f, y: %f, real: %f, desired: %f\n", row.getInput()[0], row.getInput()[1], networkOutput[0], output);
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