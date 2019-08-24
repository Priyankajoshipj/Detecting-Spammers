name := "Project"

version := "0.1"

scalaVersion := "2.11.12"

val sparkVersion = "2.4.0"

val nlpVersion = "3.9.2"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.spark" %% "spark-streaming" % sparkVersion,
  "org.apache.spark" %% "spark-hive" % sparkVersion,
  "org.scala-lang" % "scala-library" % sparkVersion,
  "edu.stanford.nlp" % "stanford-corenlp" % nlpVersion,
  "edu.stanford.nlp" % "stanford-corenlp" % nlpVersion classifier "models",
)