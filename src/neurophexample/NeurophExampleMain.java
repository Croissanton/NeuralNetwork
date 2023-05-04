package neurophexample;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.NeuralNetworkEvent;
import org.neuroph.core.events.NeuralNetworkEventListener;
import org.neuroph.nnet.Perceptron;


public class NeurophExampleMain {

    public static void main(String[] args) {

        // Create new simple perceptron network
        NeuralNetwork neuralNetwork=new Perceptron(2,1);

        // Create training set
        DataSet trainingSet=new DataSet(2,1);

        // Add training data to training set (logical OR function)
        // values for x and y must be uniformly distributed in the square of [-pi,pi]x[-pi,pi]
        for (int i = 0; i < 10; i++) {
            double x = Math.random()*2*Math.PI-Math.PI;
            double y = Math.random()*2*Math.PI-Math.PI;
            trainingSet.add(new DataSetRow (new double[] {x,y}, new double[] {Math.sin(x)*Math.cos(y)}));
        }
        // Learn the training set
        neuralNetwork.addListener(new NeuralNetworkEventListener() {

            @Override
            public void handleNeuralNetworkEvent(NeuralNetworkEvent event) {
                System.out.println(event);
            }
        });
        neuralNetwork.learn(trainingSet);
        System.out.println("Neural network trained");

        DataSet validationSet = new DataSet(2,1);
        //Add validation data to validation set
        for (int i = 0; i < 10; i++) {
            double x = Math.random()*2*Math.PI-Math.PI;
            double y = Math.random()*2*Math.PI-Math.PI;
            validationSet.add(new DataSetRow (new double[] {x,y}, new double[] {Math.sin(x)*Math.cos(y)}));
        }

        // Test the perceptron
        System.out.println("Testing trained neural network");
        for(DataSetRow dataRow : validationSet.getRows()) {
            neuralNetwork.setInput(dataRow.getInput());
            neuralNetwork.calculate();
            double[] networkOutput = neuralNetwork.getOutput();
            System.out.print("Input: " + dataRow.getInput()[0] + " " + dataRow.getInput()[1]);
            System.out.println(" Output: " + networkOutput[0]);
            if (Math.abs(networkOutput[0]-dataRow.getDesiredOutput()[0]) > 0.1) {
                System.out.println("Error: " + Math.abs(networkOutput[0]-dataRow.getDesiredOutput()[0]));
            }
        }

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