package neurophexample;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.Perceptron;


public class NeurophExampleMain {

    public static void main(String[] args) {

        // Create new simple perceptron network
        NeuralNetwork neuralNetwork=new Perceptron(2,1);

        // Create training set
        DataSet trainingSet=new DataSet(2,1);

        // Add training data to training set (logical OR function)
        trainingSet.add(new DataSetRow (new double[] {0,0}, new double[] {0}));
        trainingSet.add(new DataSetRow (new double[] {0,1}, new double[] {1}));
        trainingSet.add(new DataSetRow (new double[] {1,0}, new double[] {1}));
        trainingSet.add(new DataSetRow (new double[] {1,1}, new double[] {1}));

        // Learn the training set
        neuralNetwork.learn(trainingSet);

        // Save the trained network into file
        neuralNetwork.save("OrPerceptron.nnet");

        // Load the saved network
        NeuralNetwork neuralNetwork2 = NeuralNetwork.createFromFile("OrPerceptron.nnet");

        // Set network input
        neuralNetwork2.setInput(1,1);

        // Calculate
        neuralNetwork2.calculate();

        // Get network output
        double []networkOutput=neuralNetwork2.getOutput();

        System.out.println("The output of the network is: "+networkOutput[0]);

    }

}