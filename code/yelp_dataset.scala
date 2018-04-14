// Databricks notebook source
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType}


import org.apache.spark._
import org.apache.spark.storage._
import scala.math.Ordering
import scala.collection.JavaConversions._
import java.util.Properties
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.pipeline.{Annotation, StanfordCoreNLP}
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations
import edu.stanford.nlp.ling.CoreAnnotations
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations.SentimentAnnotatedTree
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations
import edu.stanford.nlp.util.CoreMap

import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions._
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}

// COMMAND ----------

//Accessing the dataset
val fileReviews = "/FileStore/tables/review_processed5__1_-2b66f.csv"

// COMMAND ----------

val ReviewSchema = StructType(Array(
    StructField("review_id", StringType, true),
    StructField("text", StringType, true),
    StructField("stars", IntegerType, true)))


// COMMAND ----------

val inputReviews = sqlContext.read
.format("com.databricks.spark.csv")
.option("header", "true")
.schema(ReviewSchema)
.load(fileReviews)


// COMMAND ----------

//displaying the data
inputReviews.show()

// COMMAND ----------

//dropped null values
val reviews1 = inputReviews.na.drop().cache()

// COMMAND ----------

reviews1.show()

// COMMAND ----------

//Sentiment Analysis
//This function takes each review as input, breaks the text into sentences and finds whether each sentence is positive or negative

def classify(input : String): String = {
  val regular_Exp = "\\.".r
  val filtered_input = regular_Exp.replaceAllIn(input, ". ")
  val prop: Properties = new Properties()
  prop.put("annotators", "tokenize, ssplit, parse, sentiment")
  val sentiment_pipeline: StanfordCoreNLP = new StanfordCoreNLP(prop)
  val sentiment_annotation: Annotation = new Annotation(filtered_input)
  sentiment_pipeline.annotate(sentiment_annotation)
  val sent: List[CoreMap] = sentiment_annotation.get(classOf[CoreAnnotations.SentencesAnnotation]).toList
  val counter = sent
      .map(line1 => (line1, line1.get(classOf[SentimentAnnotatedTree])))
      .map { case (line1, tree) => RNNCoreAnnotations.getPredictedClass(tree)-2 }
  if (counter.sum<0){
    return "negative"
  }
  else {
    return "positive"
  }
}

// COMMAND ----------

//Testing sentiment analysis function by passing a sample review as input
classify("Prepare to be flexible because a lot of things run out.  The food is ok but I tried it because of the lunch specials. I wanted Mac and cheese they'd ran out. Ok, sucked that up because I prob didn't need it anyway. The vegetables and rice to me was very bland. The staff is not friendly and customer service is poor.  Don't come here with a lot of expectations.")

// COMMAND ----------

//Mapping RDD to DataFrame
val reviews_RDD = reviews1.select("review_id","text").rdd.map(r=>(r.getString(0),r.getString(1)))

// COMMAND ----------

reviews_RDD.take(5)

// COMMAND ----------

//Call Sentiment Analysis function for each review

val rev_RDD1 = reviews_RDD.map{case(y,x)=>(y,classify(x))}

// COMMAND ----------

rev_RDD1.take(5)

// COMMAND ----------

val newReviewDF = rev_RDD1.toDF("review_id","sentiment").cache()

// COMMAND ----------

newReviewDF.show()

// COMMAND ----------

//Appended the classification result of reviews as last column to the dataset.
val finalReviewsdf = reviews1.join(newReviewDF, "review_id")

// COMMAND ----------

finalReviewsdf.show()

// COMMAND ----------

//Using MLlib and pipelines
val indexer = new StringIndexer()
  .setInputCol("sentiment")
  .setOutputCol("sentimentIndex")
  .fit(finalReviewsdf)
val inputData = indexer.transform(finalReviewsdf)  


// COMMAND ----------

inputData.show()

// COMMAND ----------


val assembler = new VectorAssembler()
    assembler.setInputCols(Array("stars"))
    assembler.setOutputCol("features")

// COMMAND ----------

//Logistic Regression

val lr = new LogisticRegression()
  .setMaxIter(15)
  .setRegParam(0.001)
  .setThreshold(0.4)
  .setLabelCol("sentimentIndex")
//Build pipeline
val pipeline = new Pipeline()
  .setStages(Array(assembler,lr))
//Build ParamGridBuilder
val paramGrid = new ParamGridBuilder()
  .addGrid(lr.threshold, Array(0.1,0.5,0.6))
  .addGrid(lr.regParam, Array(0.1,0.2, 0.001))
  .addGrid(lr.maxIter, Array(10,15,12))
  .build()
//Build evaluator
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("sentimentIndex")
//Build cross validator with 5 folds
val cv = new CrossValidator()
  .setEstimator(pipeline)
  .setEstimatorParamMaps(paramGrid)
  .setEvaluator(evaluator)
  .setNumFolds(30)

// COMMAND ----------

//Dividing into train and test dataset
val Array(train, test) = inputData.randomSplit(Array(0.8, 0.2))

// COMMAND ----------

//Building cross validator model
val model = cv.fit(train)
//Applying the model on test data
val result = model.transform(test)
//Selecting prediction and label columns
val predictionAndLabels = result.select("prediction", "sentimentIndex")

// COMMAND ----------

println("---------------Evaluation of Logistic Regression model ----------------------")
val evaluator_lr = new MulticlassClassificationEvaluator().setMetricName("weightedPrecision").setLabelCol("sentimentIndex")
println("Precision:" + evaluator_lr.evaluate(predictionAndLabels))
val evaluator1_lr = new MulticlassClassificationEvaluator().setMetricName("weightedRecall").setLabelCol("sentimentIndex")
println("Recall:" + evaluator1_lr.evaluate(predictionAndLabels))

// COMMAND ----------

//Accuracy using Logistic Regression

val accuracy = evaluator.evaluate(result)
println("Accuracy using Logistic Regression Model is")
println(accuracy)

// COMMAND ----------

import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
val numIterations=15
val kmeansrdd = inputData.select("stars","sentimentIndex").rdd.map(s => Vectors.dense(s.getInt(0),s.getDouble(1)))
val numClusters=Array(2,3)
for(k<-numClusters)
{
  val clusters = KMeans.train(kmeansrdd, k, numIterations)
  val WSSSE = clusters.computeCost(kmeansrdd)
  println("No.of Clusters = "+k)
println("Within Set Sum of Squared Errors for "+k+" no. of clusters  = " + WSSSE)
}
