using Microsoft.ML;
using System;
using System.Linq;

namespace WineQuality001
{
    class Program
    {
        private static readonly string _dataFilePath = @".\DataSet\winequality-red.csv";

        private static readonly string _modelFilePath = $@".\DataSet\model{DateTimeOffset.Now:yyyyMMddHmmss}.zip";

        static void Main(string[] args)
        {
            //コンテキストの生成
            MLContext mlContext = new MLContext(seed: 1);
            //データのロード
            IDataView data = mlContext.Data.LoadFromTextFile<WineQualityData>(_dataFilePath, hasHeader: true, separatorChar: ';');

            PrintDataPreview("DataSet", data);

            ////データのシャッフル
            //IDataView shuffledData = mLContext.Data.ShuffleRows(data);

            //学習データとテストデータに分割
            var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2, seed: 0);

            PrintDataPreview("TrainSet", split.TrainSet);

            PrintDataPreview("TestSet", split.TestSet);            

            //学習パイプラインの定義
            //学習データの定義  
            var dataProcessPipeline = mlContext.Transforms.Concatenate(
                    outputColumnName: "Features",
                    nameof(WineQualityData.FixedAcidity),
                    nameof(WineQualityData.VolatileAcidity),
                    nameof(WineQualityData.CitricAcid),
                    nameof(WineQualityData.ResidualSugar),
                    nameof(WineQualityData.Chlorides),
                    nameof(WineQualityData.FreeSulfurDioxide),
                    nameof(WineQualityData.TotalSulfurDioxide),
                    nameof(WineQualityData.Ph),
                    nameof(WineQualityData.Sulphates),
                    nameof(WineQualityData.Alcohol));


            PrintDataPreview("TrainSetEx", dataProcessPipeline.Preview(split.TrainSet));


            //学習アルゴリズムの定義
            var trainer = mlContext.Regression.Trainers.Sdca(labelColumnName: nameof(WineQualityData.Quality), featureColumnName: "Features");
            //var trainer = mlContext.Regression.Trainers.FastForest(labelColumnName: nameof(WineQualityData.Quality), featureColumnName: "Features");

            //学習アルゴリズムをパイプラインに設定
            var trainingPipeline = dataProcessPipeline.Append(trainer);


            Console.Write($"Training the model...\t");
            var sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            //学習データを用いて学習モデルを生成
            var trainedModel = trainingPipeline.Fit(split.TrainSet);

            sw.Stop();
            Console.WriteLine($"{sw.ElapsedMilliseconds} millisec.");

            //テストデータによるモデルの評価
            //生成した学習モデルにテストデータを設定
            IDataView predictions = trainedModel.Transform(split.TestSet);
            //学習モデルの評価
            var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: nameof(WineQualityData.Quality), scoreColumnName: "Score");

            PrintDataPreview("Evaluate", predictions);

            PrintMetrics(trainer.ToString(), metrics);            

            //学習モデルをファイルに保存
            mlContext.Model.Save(trainedModel, split.TrainSet.Schema, _modelFilePath);

            //学習モデルのロード
            ITransformer model = mlContext.Model.Load(_modelFilePath, out DataViewSchema inputSchema);
            //推論エンジンの生成
            var predictionEngine = mlContext.Model.CreatePredictionEngine<WineQualityData, WineQualityPrediction>(model);

            //疑似的にテストデータを推論のデータセットとする
            IDataView inputDataView = split.TestSet;

            WineQualityData sampleForPrediction = mlContext.Data.CreateEnumerable<WineQualityData>(inputDataView, false).First();

            //各説明変数を定義したオブジェクトを生成
            WineQualityData wineQualityData = new WineQualityData()
            {
                //TODO: 各属性の設定
                //FixedAcidity = ....
            };

            // 推論の実行
            WineQualityPrediction predictionResult = predictionEngine.Predict(sampleForPrediction);
        }

        private static void PrintDataPreview(string caption, IDataView dataView)
        {
            var dataPreview = dataView.Preview(10);

            PrintDataPreview(caption, dataPreview);
        }

        private static void PrintDataPreview(string caption, Microsoft.ML.Data.DataDebuggerPreview preview)
        {
            Console.WriteLine("==========================================");
            Console.WriteLine($"{caption}:");

            foreach (var colView in preview.ColumnView)
            {
                Console.Write($"{colView.Column.Name}, ");
            }

            Console.WriteLine();

            foreach (var rowView in preview.RowView)
            {
                foreach (var keyValue in rowView.Values)
                {
                    Console.Write($"{keyValue.Value}, ");
                }

                Console.WriteLine();
            }

            Console.WriteLine("==========================================");
        }

        private static void PrintMetrics(string caption, Microsoft.ML.Data.RegressionMetrics metrics)
        {
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*   回帰モデル: {caption} のメトリック");
            Console.WriteLine($"*");
            Console.WriteLine($"*   損失関数(LossFn): {metrics.LossFunction}");
            Console.WriteLine($"*   決定係数(R2 Score): {metrics.RSquared}");
            Console.WriteLine($"*   平均絶対誤差(Absolute loss): {metrics.MeanAbsoluteError}");
            Console.WriteLine($"*   平均二乗誤差(Squared loss): {metrics.MeanSquaredError}");
            Console.WriteLine($"*   平均二乗誤差平方根(RMS loss): {metrics.RootMeanSquaredError}");
            Console.WriteLine($"*************************************************");
        }
    }
}
