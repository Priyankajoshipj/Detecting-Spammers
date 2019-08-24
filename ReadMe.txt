Running the spark application on AWS


1. Create a s3 bucket for the project
 Refered to as <s3 location> in this file


2. Upload the .jar file to the s3 bucket
 Present inside the project folder -> target/scala-2.11/project_2.11-0.1.jar


3. Create a input folder in the s3 bucket and upload AFFIN and reviews data to the folder
      

4. Create a output folder in the s3 bucket
 Outputs of the programs will be stored in this folder under SpammerDetection folder


5. Start a cluster and use the below commands to run the spark application.
      

------------------------------------------------------------------------------------------

--
class "SpammerDetection" 

<s3 location>/project_2.11-0.1.jar

<s3 location>/input/TweetProcessing/finefoods.txt (reviews file)
<s3 location>/input/TweetProcessing/AFFIN.text (affin file)
<s3 location>/output