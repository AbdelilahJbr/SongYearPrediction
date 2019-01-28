package com.jabri.songsprediction;


import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.CollectScoresListener;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Nesterovs;

import java.io.File;

public class Main {
    private static final int nEpochs = 200;
    private static int batchSize = 1;

    public static void main(String[] args) throws Exception {
        int numberClasses = 90;

        RecordReader recordReader = new CSVRecordReader();
        recordReader.initialize(new FileSplit(new ClassPathResource("/train2.csv").getFile()));

        DataSetIterator it = new RecordReaderDataSetIterator(recordReader, batchSize, 0, numberClasses);

        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(it);
        it.setPreProcessor(normalizer);

        //Create the network
        int numInput = 90;
        int numOutputs = 90;
        int numHidden = 90;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .updater(new Nesterovs(0.01, 0.9))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInput).nOut(numHidden)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new OutputLayer.Builder()
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX)
                        .nIn(numHidden).nOut(numOutputs).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new EvaluativeListener(it,1));

        System.out.println("Training model");
        for (int i = 0; i < nEpochs; i++) {
            System.out.println("Epoch "+i);
            net.fit(it);
        }

        evaluate(net,normalizer);
        ModelSerializer.writeModel(net,new File("model"),true);
    }

    private static void evaluate(MultiLayerNetwork net,DataNormalization normalizer) throws Exception {
        RecordReader recordReader = new CSVRecordReader();
        recordReader.initialize(new FileSplit(new ClassPathResource("/test2.csv").getFile()));

        DataSetIterator it = new RecordReaderDataSetIterator(recordReader, batchSize, 0, 90);

        it.setPreProcessor(normalizer);


        System.out.println("Evaluate model....");
        Evaluation eval = new Evaluation(90); // 90 number outputs
        while(it.hasNext()){
            DataSet t = it.next();
            INDArray features = t.getFeatures();
            INDArray lables = t.getLabels();
            INDArray predicted = net.output(features,false);

            eval.eval(lables, predicted);
        }

        System.out.println(eval.stats());
    }
}