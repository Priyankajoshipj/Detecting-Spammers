import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.ml.classification.{LogisticRegression, NaiveBayes}


object SpammerDetection {

  def main(args: Array[String]): Unit = {

    if (args.length != 3) {
      println("Usage: SpammerDetection InputFile AFFINFile OutputDir")
      return
    }

    val inputFile = args(0)
    val affinFile = args(1)
    val outputDir = args(2) + "/SpammerDetection"
    var output = ""

    val spark = SparkSession.builder
      .appName("SpammerDetection")
      .master("local")
      .getOrCreate()
    val sc = spark.sparkContext
    sc.hadoopConfiguration.set("textinputformat.record.delimiter", "\n\n")

    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    val input = sc.textFile(inputFile)
      .map(line => line.split("\n"))
      .map(row => row.map(col => col.split(":").drop(1).mkString(":").trim))
      .map {
        case Array(c0, c1, c2, c3, c4, c5, c6, c7) =>
          (c0.toString, c1.toString, c2.toString, c3.split("/")(0).toString, c3.split("/")(1).toString, c4.toString, c5.toString, c6.toString, c7.toString)
        case _ => ("", "", "", "", "", "", "", "", "")
      }.toDF("productId", "userId", "profileName", "helpfulnessNumerator", "helpfulnessDenominator", "score", "time", "summary", "text")

    input.na.drop().createOrReplaceTempView("RawData")
    val rawData = spark.sql("SELECT * FROM RawData WHERE productId != ''")

    // Filter products that have less than 5 reviews
    val productsCount = rawData
      .groupBy("productId")
      .count()
      .filter($"count" > 5)

    val reviewsData = rawData.join(productsCount, "productId")

    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("tokens")
    val tokensData = tokenizer.transform(reviewsData)

    // stopWordsRemover
    val stopWordsRemover = new StopWordsRemover().setInputCol(tokenizer.getOutputCol).setOutputCol("bagOfWords").setCaseSensitive(false)
    val filteredData = stopWordsRemover.transform(tokensData)

    // Reviews View
    filteredData.createOrReplaceTempView("Reviews")
    val reviewsView = spark.sql("select * from Reviews")

    // AFFIN
    val affinData = sc.textFile(affinFile)
      .flatMap(x => x.split("\n"))
      .map(y => y.split("\t"))
      .map(x=>(x(0).toString, x(1).toInt))
    val affinMap = affinData.collectAsMap.toMap
    val broadcast = sc.broadcast(affinMap)

    // Sentiment Score
    val calculateSentimentScore = udf((words: Seq[String]) => words.map(word => broadcast.value.getOrElse(word.toLowerCase(), 0)).sum)
    val reviewsSentiment = reviewsView.withColumn("sentimentScore", calculateSentimentScore($"bagOfWords"))

    // Quartiles
    reviewsSentiment.createOrReplaceTempView("ReviewsSentiment")
    val quartiles = spark.sql("SELECT productId, percentile_approx(sentimentScore, Array(0.25,0.75)) as approxQuantile, percentile_approx(sentimentScore, 0.25) as Q1, percentile_approx(sentimentScore, 0.75) as Q3, percentile_approx(sentimentScore, 0.75)-percentile_approx(sentimentScore, 0.25) as IQR, percentile_approx(sentimentScore, 0.25)-1.5*(percentile_approx(sentimentScore, 0.75)-percentile_approx(sentimentScore, 0.25)) as lowerRange, percentile_approx(sentimentScore, 0.75)+1.5*(percentile_approx(sentimentScore, 0.75)-percentile_approx(sentimentScore, 0.25)) as upperRange FROM ReviewsSentiment GROUP BY ProductID")

    // Final Review Dataset
    val reviewsQuartile = reviewsSentiment.join(quartiles, "productId")
    val reviewsFinal = reviewsQuartile.withColumn("label", when($"sentimentScore" < $"lowerRange" or $"sentimentScore" > $"upperRange", 1).otherwise(0))
    val reviewsDataset = reviewsFinal
      .withColumn("helpfulnessNumerator", $"helpfulnessNumerator".cast(IntegerType))
      .withColumn("helpfulnessDenominator", $"helpfulnessDenominator".cast(IntegerType))
      .withColumn("score", $"score".cast(IntegerType))

    // Test Train Split
    val Array(training, test) = reviewsDataset.randomSplit(Array(0.8, 0.2), seed = 143)

    // Logistic Regression
    val assemblerLR = new VectorAssembler()
      .setInputCols(Array("helpfulnessNumerator", "helpfulnessDenominator", "score", "sentimentScore", "IQR", "lowerRange", "upperRange"))
      .setOutputCol("features")
    val lr = new LogisticRegression()
    val pipelineLR = new Pipeline().setStages(Array(assemblerLR, lr))

    // Parameter Grid - Hyper parameter tuning
    val paramGridLR = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.01, 0.05, 0.1, 0.2))
      .addGrid(lr.threshold, Array(0.2, 0.3, 0.4))
      .addGrid(lr.elasticNetParam, Array(0.7, 0.8, 0.9))
      .build()

    // Cross Validator
    val cvLR = new CrossValidator()
      .setEstimator(pipelineLR)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGridLR)
      .setNumFolds(5)
      .setParallelism(2)

    val modelLR = cvLR.fit(training)
    val resultLR = modelLR.transform(test)

    val predictionAndLabelsLR = resultLR
      .select("prediction", "label")
      .rdd.map(x => (x(0).asInstanceOf[Double], x(1).asInstanceOf[Int].toDouble))

    // Instantiate metrics object
    val metricsLR = new BinaryClassificationMetrics(predictionAndLabelsLR)

    // Overall Statistics - Logistic Regression
    output += "\nSummary Statistics for Logistic Regression\n"

    output += "\nPrecision by Threshold:\n"
    val precisionLR = metricsLR.precisionByThreshold
    precisionLR.collect.foreach { case (t, p) => output += s"Threshold: $t, Precision: $p\n" }

    output += "\nRecall by Threshold:\n"
    val recallLR = metricsLR.recallByThreshold
    recallLR.collect.foreach { case (t, r) => output += s"Threshold: $t, Recall: $r\n" }

    output += "\nF-measure by Threshold: \n"
    val f1ScoreLR = metricsLR.fMeasureByThreshold
    f1ScoreLR.collect.foreach { case (t, f) => output += s"Threshold: $t, F-score: $f, Beta = 1\n" }

    val fScoreLR = metricsLR.fMeasureByThreshold(0.5)
    fScoreLR.collect.foreach { case (t, f) => output += s"Threshold: $t, F-score: $f, Beta = 0.5\n" }

    // Area Under Precision-Recall Curve
    val auPRCLR = metricsLR.areaUnderPR
    output += s"\nArea under precision-recall curve = $auPRCLR\n"

    // AUROC
    val auROCLR = metricsLR.areaUnderROC
    output += s"\nArea under ROC = $auROCLR\n"


    // Naive Bayes
    val assemblerNB = new VectorAssembler()
      .setInputCols(Array("helpfulnessNumerator", "helpfulnessDenominator", "score", "IQR"))
      .setOutputCol("features")
    val nb = new NaiveBayes()
    val pipelineNB = new Pipeline().setStages(Array(assemblerNB, nb))

    val modelNB = pipelineNB.fit(training)
    val resultNB = modelNB.transform(test)

    val predictionAndLabelsNB = resultNB
      .select("prediction", "label")
      .rdd.map(x => (x(0).asInstanceOf[Double], x(1).asInstanceOf[Int].toDouble))

    // Instantiate metrics object
    val metricsNB = new BinaryClassificationMetrics(predictionAndLabelsNB)

    // Overall Statistics - Naive Bayes
    output += "\nSummary Statistics for Naive Bayes\n"

    output += "\nPrecision by Threshold:\n"
    val precisionNB = metricsNB.precisionByThreshold
    precisionNB.collect.foreach { case (t, p) => output += s"Threshold: $t, Precision: $p\n" }

    output += "\nRecall by Threshold:\n"
    val recallNB = metricsNB.recallByThreshold
    recallNB.collect.foreach { case (t, r) => output += s"Threshold: $t, Recall: $r\n" }

    output += "\nF-measure by Threshold: \n"
    val f1ScoreNB = metricsNB.fMeasureByThreshold
    f1ScoreNB.collect.foreach { case (t, f) => output += s"Threshold: $t, F-score: $f, Beta = 1\n" }

    val fScoreNB = metricsNB.fMeasureByThreshold(0.5)
    fScoreNB.collect.foreach { case (t, f) => output += s"Threshold: $t, F-score: $f, Beta = 0.5\n" }

    // Area Under Precision-Recall Curve
    val auPRCNB = metricsNB.areaUnderPR
    output += s"\nArea under precision-recall curve = $auPRCNB\n"

    // AUROC
    val auROCNB = metricsNB.areaUnderROC
    output += s"\nArea under ROC = $auROCNB\n"

    spark.sparkContext.parallelize(List(output)).saveAsTextFile(outputDir)
    spark.stop()
  }
}
