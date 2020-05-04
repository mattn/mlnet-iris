using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;

namespace mlnet_iris
{
    public class input
    {
        [LoadColumn(0)]
        public float SepalLength;

        [LoadColumn(1)]
        public float SepalWidth;

        [LoadColumn(2)]
        public float PetalLength;

        [LoadColumn(3)]
        public float PetalWidth;

        [LoadColumn(4)]
        public string Name; // Is not used for training
    }

    public class output
    {
        [ColumnName("PredictedLabel")]
        public uint Prediction { get; set; }

        [ColumnName("Scrore")]
        public float[] Distances { get; set; }

        public uint Label { get; set; }
    }

    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();
            var dview = context.Data.LoadFromTextFile<input>("./iris.csv", hasHeader: true, separatorChar: ',');

            var split = context.Data.TrainTestSplit(dview, testFraction: 0.2);

            var col = "Features";
            var pipeline = context.Transforms
                .Concatenate(col, "SepalLength", "SepalWidth", "PetalLength", "PetalWidth")
                .Append(context.Clustering.Trainers.KMeans(col, numberOfClusters: 3));
            var model = pipeline.Fit(split.TrainSet);

            var predictor = context.Model.CreatePredictionEngine<input, output>(model);
            var testSet = context.Data.CreateEnumerable<input>(split.TestSet, reuseRowObject: false);

            foreach (var item in testSet)
            {
                var prediction = predictor.Predict(item);
                Console.WriteLine($"{prediction.Prediction} {item.Name}");
            }
        }
    }
}
