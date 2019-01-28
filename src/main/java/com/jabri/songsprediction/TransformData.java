package com.jabri.songsprediction;

import org.apache.commons.io.FileUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.writer.RecordWriter;
import org.datavec.api.records.writer.impl.csv.CSVRecordWriter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.partition.NumberOfRecordsPartitioner;
import org.datavec.api.split.partition.Partitioner;
import org.datavec.api.transform.MathOp;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.DoubleColumnCondition;
import org.datavec.api.transform.condition.column.IntegerColumnCondition;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class TransformData {

    public static void main(String[] args) throws Exception {
        transform("train.csv","train2.csv");//Change train to trainbigdata for the large dataset
        transform("test.csv","test2.csv");
    }

    private static void transform(String in, String out) throws Exception {
        Schema.Builder builder = new Schema.Builder()
                .addColumnInteger("Year");

        for (int i = 0; i < 90; i++) {
            builder.addColumnDouble("feature" + i);
        }

        Schema inputDataSchema = builder.build();

        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
                .integerMathOp("Year", MathOp.Add, -1922)
                .build();

        //Define input and output paths:
        File inputFile = new ClassPathResource(in).getFile();
        File outputFile = new File(out);
        if (outputFile.exists()) {
            outputFile.delete();
        }
        outputFile.createNewFile();

        //Define input reader and output writer:
        RecordReader rr = new CSVRecordReader(0, ',');
        rr.initialize(new FileSplit(inputFile));

        RecordWriter rw = new CSVRecordWriter();
        Partitioner p = new NumberOfRecordsPartitioner();
        rw.initialize(new FileSplit(outputFile), p);

        //Process the data:
        List<List<Writable>> originalData = new ArrayList<List<Writable>>();
        while (rr.hasNext()) {
            originalData.add(rr.next());
        }

        List<List<Writable>> processedData = LocalTransformExecutor.execute(originalData, tp);
        rw.writeBatch(processedData);
        rw.close();


        System.out.println("DONE");
    }
}
